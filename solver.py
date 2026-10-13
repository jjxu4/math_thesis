import numpy as np


def create_initial_distribution(x_min, x_max, v_min, v_max, res_x, res_v):

    # 1D grids
    x = np.linspace(x_min, x_max, res_x)
    v = np.linspace(v_min, v_max, res_v)

    # Phase-space mesh
    X, V = np.meshgrid(x, v)

    # Initial distribution, hard coded
    f0 = ((1.0 + 0.05 * np.cos(X))* np.exp(-V**2 / 2.0) / np.sqrt(2.0 * np.pi))

    return x, v, X, V, f0

def solve_vlasov_poisson(f0, x, v, T, t_res, H=None):

    times = np.linspace(0.0, T, t_res + 1)
    dt = times[1] - times[0]

    # External field
    if H is None:
        H = np.zeros_like(x)

    f_history = np.zeros((t_res + 1, len(v), len(x)))

    f_history[0] = f0.copy()

    # Current state
    f = f0.copy()

    for n in range(t_res):
        f_star = half_x_step(f, x, v, dt)

        E = compute_E(f_star, x, v)

        f_star_star = full_v_step(f_star, x, v, E, H, dt)

        f = half_x_step(f_star_star, x, v, dt)

        f_history[n + 1] = f

    return times, f_history


# Helpers
def half_x_step(f, x, v, dt):

    dx = x[1] - x[0]
    x_left = x[0] - dx/2
    x_right = x[-1] + dx/2

    L = (x_right) - (x_left)
    X, V = np.meshgrid(x, v)

    x_eval = X - V * dt / 2.0
    x_eval = x_left + ((x_eval - x_left) % L)

    # Interpolating Stuff
    r = (x_eval - x[0]) / dx        
    k = np.floor(r).astype(int)     
    alpha = r - k
    k_above = (k + 1) % len(x)

    rows = np.arange(len(v))[:, None]

    f_below = f[rows, k]
    f_above = f[rows, k_above]

    f_return = f_below + (f_above - f_below) * alpha

    return f_return

def compute_E(f_star, x, v):

    dv = v[1] - v[0]
    dx = x[1] - x[0]

    rho = np.trapezoid(f_star, dx=dv, axis=0)
    q = 1.0 - rho

    q_hat = np.fft.fft(q)
    E_hat = np.zeros_like(q_hat)

    k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
    nonzero_mask = k != 0.0

    E_hat[nonzero_mask] = q_hat[nonzero_mask] / (1j * k[nonzero_mask])
    E = np.fft.ifft(E_hat).real

    return E

def full_v_step(f_star, x, v, E, H, dt):

    dv = v[1] - v[0]
    nv = len(v)

    X, V = np.meshgrid(x, v)

    v_eval = V + dt * (E + H)

    r = (v_eval - v[0]) / dv
    k = np.floor(r).astype(int)
    alpha = r - k
    k1 = k + 1

    validk = (k >= 0) & (k < nv)
    validk1 = (k1 >= 0) & (k1 < nv)

    f_below = np.zeros_like(f_star)
    f_above = np.zeros_like(f_star)

    i, j = np.nonzero(validk)
    f_below[i, j] = f_star[k[i, j], j]

    i, j = np.nonzero(validk1)
    f_above[i, j] = f_star[k1[i, j], j]

    f_return = f_below + (f_above - f_below) * alpha

    return f_return


