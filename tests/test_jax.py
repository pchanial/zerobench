import jax
import jax.numpy as jnp

from zerobench import Benchmark


def test_expr() -> None:
    x = jnp.array([1, 2, 3])
    y = jnp.array([2, 3, 4])
    bench = Benchmark()
    # should be transformed to '__benchfunc(x, y)' with lhs ignored and
    # __benchfunc = jax.jit(lambda x, y: x[:, None] + y)
    with bench(method='expr'):
        z = x[:, None] + y  # noqa: F841


def test_func() -> None:
    x = jnp.array([1, 2, 3])
    y = jnp.array([2, 3, 4])

    def func(x_, y_):
        return x_ + y_

    bench = Benchmark()
    with bench(method='expr'):
        # should be transformed to '__benchfunc(x, y)' with lhs ignored and
        # __benchfunc = jax.jit(func)
        z = func(x, y)  # noqa: F841


def test_func_with_arguments_as_expr() -> None:
    x = jnp.array([1, 2, 3])

    def func(x_):
        return jnp.sum(x_)

    bench = Benchmark()
    with bench(method='expr'):
        # should be transformed to '__benchfunc(x)' with
        # __benchfunc = jax.jit(lambda x: func(x + 1))
        func(x + 1)


def test_jitted_func() -> None:
    x = jnp.array([1, 2, 3])
    y = jnp.array([2, 3, 4])

    @jax.jit
    def func(x_, y_):
        return x_ + y_

    bench = Benchmark()
    with bench(method='expr'):
        func(x, y)  # should be '__benchfunc(x, y)' with __benchfunc = func


def test_jitted_func_with_arguments_as_expr() -> None:
    x = jnp.array([1, 2, 3])
    y = jnp.array([2, 3, 4])

    @jax.jit
    def func(x_, y_):
        return x_ + y_

    bench = Benchmark()
    with bench(method='expr'):
        # the fact that func is jitted is ignored, should be transformed to '__benchfunc(x, y)' with
        # __benchfunc = jax.jit(lambda x, y: func(x, y + 1))
        func(x, y + 1)
