import ast
import sys
from typing import Any

__all__ = ['CodeASTParser']


class CodeASTParser:
    """Parse and transform code for JAX benchmarking."""

    def transform_jax_code(
        self, code: str, locals: dict[str, Any], globals: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Transform code into a __benchfunc(args...) call with a jitted function.

        Args:
            code: The code to transform.
            locals: Local variables from the execution context.
            globals: Global variables from the execution context.

        Returns:
            A tuple of (new_code, new_globals) where new_code is the transformed code
            and new_globals contains the __benchfunc function.
        """
        tree = ast.parse(code, mode='exec')

        # Collect used names
        used_names = self._collect_used_names(tree, locals, globals)

        # Determine if it's a simple jitted function call
        stmt = tree.body[0]
        if isinstance(stmt, ast.Expr):
            expr = stmt.value
        elif isinstance(stmt, ast.Assign):
            expr = stmt.value
        else:
            expr = None

        jax = sys.modules['jax']
        jaxlib = sys.modules['jaxlib']
        combined = locals | globals

        if self._is_simple_jitted_call(expr, combined, jaxlib):
            # Case: func(x, y) where func is already jitted
            func_name = expr.func.id
            benchfunc = combined[func_name]
            args = [arg.id for arg in expr.args]
        elif self._is_simple_function_call(expr, combined, jaxlib):
            # Case: func(x, y) where func is not jitted
            func_name = expr.func.id
            benchfunc = jax.jit(combined[func_name])
            args = [arg.id for arg in expr.args]
        else:
            # Case: expression or call with expressions as arguments
            # Extract the expression (without lhs if assignment)
            expr_code = ast.unparse(expr)
            args = sorted(used_names)
            lambda_code = f"lambda {', '.join(args)}: {expr_code}"
            benchfunc = jax.jit(eval(lambda_code, combined))

        new_globals = combined | {'__benchfunc': benchfunc}
        new_code = f"__benchfunc({', '.join(args)})"
        return new_code, new_globals

    def _collect_used_names(
        self, tree: ast.AST, locals: dict[str, Any], globals: dict[str, Any]
    ) -> set[str]:
        """Collect variable names used in the AST."""

        class NameCollector(ast.NodeVisitor):
            def __init__(self):
                self.names: set[str] = set()

            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Load):
                    self.names.add(node.id)
                self.generic_visit(node)

        collector = NameCollector()
        collector.visit(tree)

        # Filter: keep only names in locals/globals, exclude builtins and callables
        combined = locals | globals
        return {
            name for name in collector.names if name in combined and not callable(combined[name])
        }

    def _is_simple_jitted_call(
        self, expr: ast.expr | None, combined: dict[str, Any], jaxlib: Any
    ) -> bool:
        """Check if it's a func(x, y, ...) call where func is jitted and all args are Names."""
        if not isinstance(expr, ast.Call):
            return False
        if not isinstance(expr.func, ast.Name):
            return False
        func = combined.get(expr.func.id)
        if not isinstance(func, jaxlib._jax.PjitFunction):
            return False
        return all(isinstance(arg, ast.Name) for arg in expr.args) and not expr.keywords

    def _is_simple_function_call(
        self, expr: ast.expr | None, combined: dict[str, Any], jaxlib: Any
    ) -> bool:
        """Check if it's a func(x, y, ...) call where all args are Names."""
        if not isinstance(expr, ast.Call):
            return False
        if not isinstance(expr.func, ast.Name):
            return False
        func = combined.get(expr.func.id)
        if func is None or isinstance(func, jaxlib._jax.PjitFunction):
            return False
        return all(isinstance(arg, ast.Name) for arg in expr.args) and not expr.keywords
