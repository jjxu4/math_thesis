import numpy as np
import matplotlib.pyplot as plt

from solver import solve_vlasov_poisson
from optimizer import compute_C, solve_adjoint


def optimizer(f_desired, a_initial_guess, modes, f0, x, v, T, t_res, tolerance=1e-6, learning_rate=1.0, max_iter=1000, showDiag=True):

    dx = x[1] - x[0]
    dv = v[1] - v[0]
    x_left = x[0] - dx / 2.0
    L = len(x) * dx

    a = np.array(a_initial_guess, dtype=float, copy=True)
    modes = np.asarray(modes)

    basis = np.cos(0.5 * x[:, None] * modes)
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

        if showDiag == True:
            print("=====================================")
            print(f"Iteration {iteration}/{max_iter}")
            print("Cost function: J(f[H(a1, ..., ak)]) =", J)
            print("a1, ..., ak =", a)
            print(f"gradient norm = {gradient_norm}")
            print(" ")

        # Stopping conditions
        if gradient_norm <= tolerance:
            return a, H, f_history[-1]

        if iteration == max_iter:
            break

        a = a - learning_rate * gradient
        

    # Could not finish optimization
    raise RuntimeError(
        f"Tolerance was not reached after {max_iter} updates. "
        f"Gradient norm: {gradient_norm:.6e}. Objective: {J:.6e}"
    )

def scan_1d(f_desired, a_min, a_max, a_step, modes, f0, x, v, T, t_res):

    modes = np.asarray(modes)

    dx = x[1] - x[0]
    dv = v[1] - v[0]
    n_steps = int(np.floor((a_max - a_min) / a_step + 1e-12))
    a_values = a_min + a_step * np.arange(n_steps + 1)
    J_values = np.zeros(len(a_values))
    basis = np.cos(0.5 * x * modes[0])

    for i, a in enumerate(a_values):
        H = a * basis
        times, f_history = solve_vlasov_poisson(f0, x, v, T, t_res, H)
        J_values[i] = 0.5 * dx * dv * np.sum((f_history[-1] - f_desired)**2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(a_values, J_values)
    ax.set_xlabel("a")
    ax.set_ylabel("J")
    fig.tight_layout()
    plt.show()


def scan_2d(f_desired, a1_min, a1_max, a2_min, a2_max, a1_step, a2_step, modes, f0, x, v, T, t_res):

    modes = np.asarray(modes)

    dx = x[1] - x[0]
    dv = v[1] - v[0]
    n1_steps = int(np.floor((a1_max - a1_min) / a1_step + 1e-12))
    n2_steps = int(np.floor((a2_max - a2_min) / a2_step + 1e-12))
    a1_values = a1_min + a1_step * np.arange(n1_steps + 1)
    a2_values = a2_min + a2_step * np.arange(n2_steps + 1)

    if len(a1_values) < 2 or len(a2_values) < 2:
        raise ValueError("Use at least two samples for each coefficient.")

    J_values = np.zeros((len(a2_values), len(a1_values)))
    basis = np.cos(0.5 * x[:, None] * modes)

    for i, a2 in enumerate(a2_values):
        for j, a1 in enumerate(a1_values):
            H = basis @ np.array([a1, a2])
            times, f_history = solve_vlasov_poisson(f0, x, v, T, t_res, H)
            J_values[i, j] = 0.5 * dx * dv * np.sum((f_history[-1] - f_desired)**2)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(a1_values, a2_values, J_values, shading="auto", cmap="inferno")
    ax.set_xlabel(f"a{modes[0]}")
    ax.set_ylabel(f"a{modes[1]}")
    fig.colorbar(mesh, ax=ax, label="J")
    fig.tight_layout()
    plt.show()
