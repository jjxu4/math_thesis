import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def solve_vlasov_poisson(f0, x, v, T, t_res, H=None, returnEhist=False):

    times = np.linspace(0.0, T, t_res + 1)
    dt = times[1] - times[0]

    # External field
    if H is None:
        H = np.zeros_like(x)

    f_history = np.zeros((t_res + 1, len(v), len(x)))       # t_res + 1 grids, each grid has len(v) arrays, each with len(x) elements

    f_history[0] = f0.copy()                # f0 should be a 2D array, first index representing velocity second index representing position

    # Current state
    f = f0.copy()
    E_hist = np.zeros((t_res + 1, len(x)))
    if returnEhist:
        E_hist[0] = compute_E(f, x, v)

    for n in range(t_res):
        f_star = half_x_step(f, x, v, dt)

        E = compute_E(f_star, x, v)

        f_star_star = full_v_step(f_star, x, v, E, H, dt)

        f = half_x_step(f_star_star, x, v, dt)

        f_history[n + 1] = f
        if returnEhist:
            E_hist[n + 1] = compute_E(f, x, v)

    to_return = (times, f_history)
    if returnEhist == True:
        to_return = (times, f_history, E_hist)

    return to_return

def plot_vp_solution(f0, x, v, T, t_res, H=None):

    if H is None:
        H = np.zeros_like(x)

    times, f_history, E_hist = solve_vlasov_poisson(f0, x, v, T, t_res, H, returnEhist=True)
    total_hist = E_hist + H
    dx = x[1] - x[0]
    electric_energy = dx * np.sum(E_hist**2, axis=1)
    plt.rcParams["animation.embed_limit"] = 100.0

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.08, right=0.68, bottom=0.49, top=0.92)

    mesh = ax.pcolormesh(x, v, f_history[0], shading="auto", cmap="viridis", vmin=0.0, vmax=f_history.max())
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(v[0], v[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title(f"t = {times[0]:.1f}")

    colorbar_ax = fig.add_axes([0.08, 0.39, 0.60, 0.025])
    fig.colorbar(mesh, cax=colorbar_ax, orientation="horizontal", label="f(x, v)")

    ax_E = ax.twinx()
    ax_H = ax.twinx()
    ax_total = ax.twinx()

    for axis, values, color, label, offset in [
        (ax_E, E_hist, "tab:blue", "E(x)", 0),
        (ax_H, H, "tab:orange", "H(x)", 65),
        (ax_total, total_hist, "tab:green", "E(x) + H(x)", 130),
    ]:
        axis.spines["right"].set_position(("outward", offset))
        axis.spines["right"].set_color(color)
        axis.set_ylabel(label, color=color)
        axis.tick_params(axis="y", colors=color)
        limit = np.max(np.abs(values))
        if limit == 0.0:
            limit = 1.0
        axis.set_ylim(-1.1 * limit, 1.1 * limit)

    line_E, = ax_E.plot(x, E_hist[0], color="tab:blue")
    line_H, = ax_H.plot(x, H, color="tab:orange")
    line_total, = ax_total.plot(x, total_hist[0], color="tab:green")

    ax_energy = fig.add_axes([0.08, 0.09, 0.60, 0.19])
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel(r"$\int |E|^2\,dx$")
    ax_energy.set_xlim(times[0], times[-1])
    energy_limit = electric_energy.max()
    if energy_limit == 0.0:
        energy_limit = 1.0
    ax_energy.set_ylim(0.0, 1.1 * energy_limit)
    line_energy, = ax_energy.plot(times, electric_energy, color="tab:blue")
    time_bar = ax_energy.axvline(times[0], color="red")

    def update(frame):
        mesh.set_array(f_history[frame].ravel())
        line_E.set_ydata(E_hist[frame])
        line_total.set_ydata(total_hist[frame])
        ax.set_title(f"t = {times[frame]:.1f}")
        time_bar.set_xdata([times[frame], times[frame]])
        return mesh, line_E, line_H, line_total, line_energy, time_bar

    anim = FuncAnimation(fig, update, frames=len(times), interval=100)
    plt.close(fig)

    return anim

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

    rho = dv * np.sum(f_star, axis=0)
    q = 1.0 - rho

    q_hat = np.fft.fft(q)
    E_hat = np.zeros_like(q_hat)

    k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
    nonzero_mask = (k != 0.0)

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


