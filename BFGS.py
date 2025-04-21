import jax
import jax.numpy as jnp

def backtracking_line_search(f, x, p, grad_x, alpha=1.0, rho=0.8, c=1e-1):
    """
    Simple backtracking line search using the Armijo condition.
    
    f: objective function.
    x: current point.
    p: search direction.
    grad_x: gradient at x.
    alpha: initial step size.
    rho: factor to decrease alpha.
    c: Armijo constant.
    """
    # Evaluate the current function value.
    fx = f(x)
    # Iteratively decrease alpha until the Armijo condition is met.
    while f(x + alpha * p) > fx + c * alpha * jnp.dot(grad_x, p):
        alpha *= rho
    return alpha

def bfgs(f, x0, max_iter=100, tol=1e-6):
    """
    BFGS optimization using JAX.
    
    f: function to minimize.
    x0: initial point as a JAX array.
    max_iter: maximum iterations.
    tol: tolerance for stopping criterion (based on gradient norm).
    """
    # Current point and dimension.
    x = x0
    n = x0.shape[0]
    # Initialize the inverse Hessian approximation as the identity matrix.
    H = jnp.eye(n)
    # Function to compute the gradient using automatic differentiation.
    grad_f = jax.grad(f)
    
    for i in range(max_iter):
        g = grad_f(x)
        # Check for convergence.
        if jnp.linalg.norm(g) < tol:
            print(f"Convergence reached at iteration {i}")
            break
        
        # Compute the search direction.
        p = -H @ g
        # Perform a line search to determine step size.
        alpha = backtracking_line_search(f, x, p, g)
        
        # Update step.
        s = alpha * p
        x_new = x + s
        y = grad_f(x_new) - g
        
        # Compute scaling factor.
        rho = 1.0 / jnp.dot(y, s)
        
        # BFGS Hessian update.
        I = jnp.eye(n)
        H = (I - rho * jnp.outer(s, y)) @ H @ (I - rho * jnp.outer(y, s)) + rho * jnp.outer(s, s)
        
        # Move to the next point.
        x = x_new
    return x

# Example usage with the Rosenbrock function.
def rosenbrock(x):
    """The Rosenbrock function in 2D."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Initial guess.
x0 = jnp.array([-.2, .0])
opt_x = bfgs(rosenbrock, x0)

print("Optimized x:", opt_x)
