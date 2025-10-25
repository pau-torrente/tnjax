import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu, solve_triangular, inv


@jax.jit
def maxvol(A: jnp.ndarray, tol: float = 1.05, max_iter: int = 100):
    """
    Optimized JAX Maxvol algorithm (JIT compiled) for selecting rows forming
    a submatrix with maximal volume.

    Args:
        A: Input matrix of shape (n, r), with n >= r.
        tol: Tolerance for convergence (default 1.05).
        max_iter: Maximum number of iterations.

    Returns:
        index: Array of r selected row indices.
        D: The final coefficient matrix.
    """
    n, r = A.shape
    assert n >= r, "Number of rows must be >= number of columns"

    # Step 1: LU decomposition with pivoting
    P, L, U = lu(A)
    index = jnp.argmax(P, axis=1)

    # Step 2: Initial r×r submatrix
    C = U[:r, :]
    D = solve_triangular(C.T, A.T, lower=True).T  # shape (n, r)

    # Step 3: Iterative maxvol update
    def cond_fun(val):
        D, index, iter = val
        i, j = jnp.unravel_index(jnp.argmax(jnp.abs(D)), D.shape)
        return (jnp.abs(D[i, j]) > tol) & (iter < max_iter)

    def body_fun(val):
        D, index, iter = val
        i, j = jnp.unravel_index(jnp.argmax(jnp.abs(D)), D.shape)
        coeff = -1.0 / D[i, j]

        tmp_row = D[i].copy()
        tmp_column = D[:, j].copy()
        tmp_column = tmp_column.at[i].add(-1.0)
        tmp_row = tmp_row.at[j].add(1.0)

        D = D + coeff * jnp.outer(tmp_column, tmp_row)
        index = index.at[i].set(j)
        iter += 1
        return D, index, iter

    D, index, _ = jax.lax.while_loop(cond_fun, body_fun, (D, index, 0))
    return index[:r], D


@jax.jit
def greedy_pivot_finder(
    A: jnp.ndarray,
    I_1i: jnp.ndarray,
    current_I_pos: jnp.ndarray,
    J_1j: jnp.ndarray,
    current_J_pos: jnp.ndarray,
    tol: float = 1e-10,
):
    """
    Greedy pivot finder algorithm implemented in JAX.
    """
    # Extract current pivot submatrix (core)
    square_core = A[current_I_pos[:, None], current_J_pos]

    # Compute the cross approximation
    Approx = A[:, current_J_pos] @ inv(square_core) @ A[current_I_pos, :]

    # Find pivot with maximum residual
    i_new, j_new = jnp.unravel_index(jnp.argmax(jnp.abs(A - Approx)), A.shape)

    # Compute residual at pivot
    residual = jnp.abs(A[i_new, j_new] - Approx[i_new, j_new])

    # Check stopping conditions
    i_in = jnp.any(current_I_pos == i_new)
    j_in = jnp.any(current_J_pos == j_new)
    stop = (i_in | j_in) | (residual < tol)

    def stop_case(_):
        I = I_1i[current_I_pos]
        J = J_1j[current_J_pos]
        err = jnp.sum(jnp.abs(A - Approx))
        return I, J, len(I), len(J), err

    def continue_case(_):
        new_I_pos = jnp.concatenate([current_I_pos, jnp.array([i_new])])
        new_J_pos = jnp.concatenate([current_J_pos, jnp.array([j_new])])
        I_new = I_1i[new_I_pos]
        J_new = J_1j[new_J_pos]
        err = jnp.sum(jnp.abs(A - Approx))
        return I_new, J_new, len(I_new), len(J_new), err

    return jax.lax.cond(stop, stop_case, continue_case, operand=None)
