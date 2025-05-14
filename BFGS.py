import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from test_functions import *


def backtracking_line_search(phi, phi_0, grad_phi_0, a_max, rho=0.8, c=9e-1):
    """
    Simple backtracking line search using the Armijo condition.

    phi: objective function.
    x: current point.
    p: search direction.
    grad_x: gradient at x.
    alpha: initial step size.
    rho: factor to decrease alpha.
    c: Armijo constant.
    """
    a = a_max
    while phi(a) > phi_0 + c * a * grad_phi_0:
        a *= rho
    return a


def line_search(
    phi,
    phi_0,
    grad_phi_0,
    a_max,
    zoom_method="strong_wolfe",
    c1=1e-4,
    c2=9e-1,
    max_iter=100,
):
    a = a_max / 2.0
    a_prev = 0.0
    i = 0
    phi_curr = phi_0
    zoom = None
    if zoom_method == "strong_wolfe":
        zoom = lambda al, au: interpolation_line_search(
            phi, phi_0, grad_phi_0, al, au, c1=c1, c2=c2, max_iter=10
        )
    elif zoom_method == "armijo":
        return backtracking_line_search(
            phi, phi_0, grad_phi_0, a_max=a_max, rho=0.5, c=c2
        )
    else:
        raise ValueError("Zoom method not implemented")

    while i < max_iter:
        phi_prev = phi_curr
        phi_curr = phi(a)
        if phi_curr > phi_0 + c1 * a * grad_phi_0 or (phi_curr >= phi_prev and i > 0):
            return zoom(a_prev, a)
        grad_phi_curr = jax.grad(phi)(a)
        if jnp.abs(grad_phi_curr) <= -1 * c2 * grad_phi_0:
            return a
        if grad_phi_curr >= 0:
            return zoom(a_prev, a)
        a_prev = a
        a = (a + a_max) / 2.0
        i += 1
    print("Line Search Failed")
    return 0.0


def gradient_descent(
    f, x0,lr=0.001 , max_iter=1000,  tol=1e-15
):
    """
    Gradient descent optimization 

    f: function to minimize.
    x0: initial point as a JAX array.
    max_iter: maximum iterations.
    tol: tolerance for stopping criterion (based on gradient norm).
    """
    # Current point and dimension.
    x = x0
    n = x0.shape[0]
    # Function to compute the gradient using automatic differentiation.
    grad_f = jax.grad(f)
    xs = [x0]
    i = 0
    g = grad_f(x0)
    while jnp.linalg.norm(g) > tol and i < max_iter:
        s = -lr * g
        x_new = x + s
        # Move to the next point.
        x = x_new
        xs.append(x)
        g = grad_f(x)
        i += 1
    if i == max_iter:
        print("Max iterations reached")
    return x, jnp.array(xs)

def steepest_descent(
    f, x0,  max_iter=1000,line_search_max_iter=20, tol=1e-15
):
    """
    Steepest descent optimization.

    f: function to minimize.
    x0: initial point as a JAX array.
    max_iter: maximum iterations.
    tol: tolerance for stopping criterion (based on gradient norm).
    """
    # Current point and dimension.
    x = x0
    n = x0.shape[0]
    # Function to compute the gradient using automatic differentiation.
    grad_f = jax.grad(f)
    xs = [x0]
    i = 0
    g = grad_f(x0)
    while jnp.linalg.norm(g) > tol and i < max_iter:
        # Compute the line search direction.
        p = -g
        # Perform a line search to determine step size.
        phi = lambda alpha: f(x + alpha * p)
        grad_phi_0 = jnp.dot(g, p)
        alpha = line_search(phi, phi(0.0), grad_phi_0, 1.0,max_iter=line_search_max_iter)
        # Update step.
        s = alpha * p
        x_new = x + s
        # Move to the next point.
        x = x_new
        xs.append(x)
        g = grad_f(x)
        i += 1

    if i == max_iter:
        print("Max iterations reached")
    return x, jnp.array(xs)

def interpolation_line_search(
    phi, phi_0, grad_phi_0, al, au, c1=1e-4, c2=9e-1, max_iter=30
):
    i = 0
    a = al
    a_prev = 0.0
    while i < max_iter:
        if jnp.abs(al) <= 1e-14:
            a = (-grad_phi_0 * au**2) / (2.0 * (phi(au) - phi_0 - au * grad_phi_0))
        else:
            if jnp.linalg.norm(au - al) < 1e-14:
                a = (al + au) / 2.0 
            else:
                coeff = (1.0 / (al**2 * au**2 * (au - al))) * (
                    jnp.array([[al**2, -(au**2)], [-(al**3), au**3]])
                    @ jnp.array(
                        [
                            phi(au) - phi_0 - au * grad_phi_0,
                            phi(al) - phi_0 - al * grad_phi_0,
                        ]
                    )
                )
                a_poly, b_poly = coeff[0], coeff[1]
                if jnp.abs(a_poly) < 1e-14 or ((b_poly**2 - 3 * a_poly * grad_phi_0) < 0):
                    a = (al + au) / 2.0
                else:
                    a = (-b_poly + jnp.sqrt(b_poly**2 - 3 * a_poly * grad_phi_0)) / (
                        3 * a_poly
                    )
        if jnp.abs(a - a_prev) < 1e-10:
            a = a_prev / 2.0
        curr_phi = phi(a)
        if curr_phi > phi_0 + c1 * a * grad_phi_0 or (curr_phi >= phi(al) and i > 0):
            au = a
        else:
            grad_phi_curr = jax.grad(phi)(a)
            if jnp.abs(grad_phi_curr) <= -1.0 * c2 * grad_phi_0:
                return a
            if grad_phi_curr * (au - al) >= 0:
                au = al
            al = a
        i += 1
    return al


def bfgs(
    f, x0, zoom_method="strong_wolfe", max_iter=100, line_search_max_iter=100, tol=1e-15
):
    """
    BFGS optimization. 

    f: function to minimize.
    x0: initial point as a JAX array.
    max_iter: maximum iterations.
    tol: tolerance for stopping criterion (based on gradient norm).
    """
    # Current point and dimension.
    x = x0
    n = x0.shape[0]
    # Initialize the inverse Hessian approximation as the identity matrix.
    Hinv = jnp.eye(n)
    # Function to compute the gradient using automatic differentiation.
    grad_f = jax.grad(f)
    xs = [x0]
    i = 0
    r = jnp.inf
    g = grad_f(x0)
    while jnp.linalg.norm(g) > tol and i < max_iter:

        # Compute the search direction.
        p = -Hinv @ g
        # Perform a line search to determine step size.
        phi = lambda alpha: f(x + alpha * p)
        grad_phi_0 = jnp.dot(g, p)
        alpha = line_search(phi, phi(0.0), grad_phi_0, 1.0, zoom_method=zoom_method)
        # Update step.
        s = alpha * p
        x_new = x + s
        y = grad_f(x_new) - g
        # Compute scaling factor.
        rho =  jnp.dot(y, s)

        # BFGS Hessian update.
        Hinv = (
            Hinv
            + ((jnp.dot(s, y) + y.T @ (Hinv @ y)) * (jnp.outer(s, s)))
            / (jnp.dot(s, y) ** 2)
            - (jnp.outer(Hinv @ y, s) + jnp.outer(s, jnp.dot(y.T, Hinv)))
            / (jnp.dot(s, y))
        )

        # Move to the next point.
        r = f(x) - f(x_new)
        if zoom_method == "strong_wolfe" and jnp.abs(r) < tol:
            break

        x = x_new
        xs.append(x)
        g = grad_f(x)
        i += 1
    if i == max_iter:
        print("Max iterations reached")
    return x, jnp.array(xs)


def lbfgs(f, x0, history_size=10, max_iter=100,zoom_method="strong_wolfe", tol=1e-15):
    """
    Limited-memory BFGS optimization. 

    f: function to minimize.
    x0: initial point as a JAX array.
    history_size: number of previous gradients and steps to store.
    max_iter: maximum iterations.
    tol: tolerance for stopping criterion (based on gradient norm).
    """
    x= x0
    n = x0.shape[0]
    # Function to compute the gradient using automatic differentiation.
    grad_f = jax.grad(f)
    xs = [x0]
    i = 0
    g = grad_f(x0)
    s_list = []
    y_list = []
    alpha_list = []
    while jnp.linalg.norm(g) > tol and i < max_iter:
        # Compute the search direction.
        q = g

        for i in range(len(s_list)): 
            # alpha = jnp.dot(s, q) / jnp.dot(y, s)
            alpha_list[i] = jnp.dot(s_list[i], q) / jnp.dot(y_list[i], s_list[i])
            q -= alpha_list[i] * y_list[i]
        if len(s_list) == 0:
            # If no history, use the identity matrix.
            gamma = 1.0
        else:
            gamma = jnp.dot(y_list[-1], s_list[-1]) / jnp.dot(y_list[-1], y_list[-1])


        # gamma = jnp.dot(s_list[-1], y_list[-1])/ jnp.dot(y_list[-1], y_list[-1])
        H0 = gamma * jnp.eye(n)
        r = H0 @ q
        for s,y,alpha in zip(reversed(s_list), reversed(y_list), reversed(alpha_list)):
            beta = jnp.dot(y, r) / jnp.dot(y, s)
            r += (alpha - beta) * s
        p = -r

        # Perform a line search to determine step size.
        phi = lambda alpha: f(x + alpha * p)
        grad_phi_0 = jnp.dot(g, p)
        alpha = line_search(phi, phi(0.0), grad_phi_0, 1.0,zoom_method=zoom_method)
        # Update step.
        s = alpha * p
        x_new = x + s
        y = grad_f(x_new) - g
        # Store the last history_size steps and gradients.
        if len(s_list) == history_size:
            s_list.pop(0)
            y_list.pop(0)
            alpha_list.pop(0)

        s_list.append(s)
        y_list.append(y)
        alpha_list.append(0)
        # alpha_list.append(alpha)

        # Move to the next point.
        x = x_new
        xs.append(x)
        g = grad_f(x)
        i += 1
    if i == max_iter:
        print("Max iterations reached")
    return x, jnp.array(xs)




def newton(f, x0,  max_iter=100, tol=1e-15):
    """
    Newton's method for optimization.
    """
    x = x0
    n = x0.shape[0]
    grad_f = jax.grad(f)
    hess_f = jax.hessian(f)
    xs = [x0]
    i = 0
    r = jnp.inf
    g = grad_f(x0)

    while jnp.abs(r) > tol and jnp.linalg.norm(g) > tol and i < max_iter:

        # Compute the search direction.
        p = -jnp.linalg.solve(hess_f(x), g)
        # Perform a line search to determine step size.
        phi = lambda alpha: f(x + alpha * p)
        grad_phi_0 = jnp.dot(g, p)
        alpha = backtracking_line_search(phi, phi(0.0), grad_phi_0, 1.0)
        # Update step.
        s = alpha * p
        x_new = x + s
        r = f(x) - f(x_new)
        # Move to the next point.
        x = x_new
        xs.append(x)
        g = grad_f(x)
        i += 1
    if i == max_iter:
        print("Max iterations reached")

    return x, jnp.array(xs)


if __name__ == "__main__":
    # Example usage with the Rosenbrock function.
    def rosenbrock(x):
        """The Rosenbrock function in 2D."""
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    f = rosenbrock
    # Initial guess.
    x0 = jnp.array([-.5, -.5])
    # opt_x, xs = gradient_descent(f, x0, max_iter=10000, tol=1e-14)
    # opt_x, xs = lbfgs(
    #     f,
    #     x0,
    #     history_size=5,
    #     max_iter=1000,
    #     tol=1e-14,
    # )
    opt_x, xs = steepest_descent(
        f,
        x0,
        max_iter=1000,
        tol=1e-14,
    )

    print("Optimized x:", opt_x, f(opt_x), jnp.linalg.norm(jax.grad(f)(opt_x)))
    print("Iterations:", xs.shape[0] - 1)
