import numpy as np

from solver import solve_vlasov_poisson, half_x_step, full_v_step, compute_E


def optimizer(f_desired, H_initial_guess, f0, x, v, T, t_res, tolerance=1e-6, learning_rate=1.0, max_iter=1000):


    dx = x[1] - x[0]
    dv = v[1] - v[0]
    H = np.array(H_initial_guess, dtype=float, copy=True)
    C = compute_C(x)

    for iteration in range(max_iter + 1):
        times, f_history = solve_vlasov_poisson(f0, x, v, T, t_res, H)

        g = f_history[-1] - f_desired
        sigma_history = solve_adjoint(g, f_history, x, v, T, H, C)
        gradient = dx * dv * np.sum(sigma_history, axis=0)
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm < tolerance:
            return H, f_history[-1]

        if iteration == max_iter:
            break

        H = H - learning_rate * gradient

    raise RuntimeError(
        f"Tolerance was not reached after {max_iter} updates. "
        f"Gradient norm: {gradient_norm:.6e}"
    )


def compute_C(x):

    dx = x[1] - x[0]
    nx = len(x)

    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    nonzero_mask = k != 0.0

    C_hat = np.fft.fft(np.eye(nx), axis=0)
    C_hat[nonzero_mask] = C_hat[nonzero_mask] / (1j * k[nonzero_mask, None])
    C_hat[~nonzero_mask] = 0.0
    C = np.fft.ifft(C_hat, axis=0).real

    return C


def solve_adjoint(g, f_history, x, v, T, H, C):

    t_res = len(f_history) - 1
    dt = T / t_res
    sigma_history = np.zeros((t_res, len(x)))

    for n in range(t_res - 1, -1, -1):
        f_star = half_x_step(f_history[n], x, v, dt)
        E = compute_E(f_star, x, v)

        g_star_star = half_x_step(g, x, v, -dt)
        g_star, sigma = adjoint_v_step(g_star_star, f_star, x, v, E, H, dt, C)
        sigma_history[n] = sigma

        if n >= 1:
            g = half_x_step(g_star, x, v, -dt)

    return sigma_history


def adjoint_v_step(g, f_star, x, v, E, H, dt, C):

    dv = v[1] - v[0]
    nv = len(v)

    X, V = np.meshgrid(x, v)
    v_eval = V + dt * (E + H)

    r = (v_eval - v[0]) / dv
    k = np.floor(r).astype(int)
    k1 = k + 1

    validk = (k >= 0) & (k < nv)
    validk1 = (k1 >= 0) & (k1 < nv)

    f_below = np.zeros_like(f_star)
    f_above = np.zeros_like(f_star)

    i, j = np.nonzero(validk)
    f_below[i, j] = f_star[k[i, j], j]

    i, j = np.nonzero(validk1)
    f_above[i, j] = f_star[k1[i, j], j]

    sigma = (dt / dv) * np.sum(g * (f_above - f_below), axis=0)

    weights = np.full(nv, dv)
    weights[0] = dv / 2.0
    weights[-1] = dv / 2.0

    g_return = full_v_step(g, x, v, E, H, -dt)
    g_return = g_return - weights[:, None] * (C.T @ sigma)[None, :]

    return g_return, sigma
