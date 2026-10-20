import numpy as np

from solver import solve_vlasov_poisson
from optimizer import compute_C, solve_adjoint


def optimizer(f_desired, a_initial_guess, modes, f0, x, v, T, t_res, tolerance=1e-6, learning_rate=1.0, max_iter=1000):

    dx = x[1] - x[0]
    dv = v[1] - v[0]
    x_left = x[0] - dx / 2.0
    L = len(x) * dx

    a = np.array(a_initial_guess, dtype=float, copy=True)
    modes = np.asarray(modes)
    basis = np.sin(2.0 * np.pi * (x[:, None] - x_left) * modes[None, :] / L)
    C = compute_C(x)

    for iteration in range(max_iter + 1):
        H = basis @ a
        times, f_history = solve_vlasov_poisson(f0, x, v, T, t_res, H)

        J = 0.5 * dx * dv * np.sum((f_history[-1] - f_desired)**2)
        g = f_history[-1] - f_desired
        sigma_history = solve_adjoint(g, f_history, x, v, T, H, C)

        gradient_H = dx * dv * np.sum(sigma_history, axis=0)
        gradient = basis.T @ gradient_H
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm <= tolerance:
            return a, H, f_history[-1]

        if iteration == max_iter:
            break

        a = a - learning_rate * gradient

    raise RuntimeError(
        f"Tolerance was not reached after {max_iter} updates. "
        f"Gradient norm: {gradient_norm:.6e}. Objective: {J:.6e}"
    )
