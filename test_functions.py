import jax
import jax.numpy as jnp


def f1(x):
    """
    Adjiman Function
    """
    return jnp.cos(x[0]) * jnp.sin(x[1]) - (x[0] / (x[1] ** 2 + 1))


def f2(x):
    """
    Rosenbrock N-D
    """
    return jnp.sum(100 * (x[1:] - x[:-1]) ** 2 + (x[:-1] - 1) ** 2)


def f3(x):
    """
    Paviani Function
    """
    return jnp.sum(jnp.log(x - 2) ** 2 + jnp.log(10 - x) ** 2) - jnp.prod(x) ** (0.2)


def f4(x):
    """
    Csendes Function
    """
    return jnp.sum(x**6 * (2 + jnp.sin(1.0 / x)))


def f5(x):
    """
    Griewank Function
    """
    return (
        jnp.sum(x**2)
        - jnp.prod(jnp.cos(2 * jnp.pi * x / jnp.sqrt(jnp.arange(1, x.shape[0] + 1))))
        + 1
    )


def f6(x):
    """
    Hosaki Function
    """
    return (
        (1 - 8 * x[0] + 7 * x[0] ** 2 - (7 / (3 * x[0] ** 2)) + (1 / (4 * x[0] ** 4)))
        * x[1] ** 2
        * jnp.exp(-x[1])
    )


def f7(x):
    """
    Brent Function
    """
    return (x[0] + 10) ** 2 + (x[1] + 10) ** 2 + jnp.exp(-x[0] ** 2 - x[1] ** 2)


def f8(x):
    """
    Giunta Function
    """
    return 0.6 + jnp.sum(
        jnp.sin((16.0 / 15.0) * x - 1)
        + jnp.sin((16.0 / 15.0) * x - 1) ** 2
        + jnp.sin(4 * ((16.0 / 15.0) * x - 1))
    )


def f9(x):
    """
    Styblinski-Tang Function
    """
    return 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x)


def f10(x):
    """
    Trid 6 Function
    """
    return jnp.sum((x - 1) ** 2) - jnp.sum(x[1:] * x[:-1])


def f11(x):
    """
    Trefethen Function
    """
    return (
        jnp.exp(jnp.sin(50 * x[0]))
        + jnp.sin(60 * jnp.exp(x[1]))
        + jnp.sin(70 * jnp.sin(x[0]))
        + jnp.sin(jnp.sin(80 * x[1]))
        - jnp.sin(10 * (x[0] + x[1]))
        + 0.25 * (x[0] ** 2 + x[1] ** 2)
    )


def f12(x):
    """
    Ursem 1 Function
    """
    return -jnp.sin(2 * x[0] - 0.5 * jnp.pi) - 3 * jnp.cos(x[1]) - 0.5 * x[0]


def f13(x):
    """
    Wolfe Function
    """
    return (4.0 / 3) * (x[0] ** 2 + x[1] ** 2 - x[0] * x[1]) ** (0.75) + x[2]
