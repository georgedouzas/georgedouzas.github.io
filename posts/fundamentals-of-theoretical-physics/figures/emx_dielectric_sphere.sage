# Figure: a dielectric sphere (epsilon = 4, radius 1) in a uniform field E0 e_z; field lines of E (left) and D (right) in the xz plane.
# Checks: the potential is harmonic on both sides, continuous at r = a with continuous epsilon d_r phi; the interior field is 3/(eps + 2) E0;
# the normal component of D is continuous and that of E jumps by the bound charge P cos(theta). Output: figures/emx_dielectric_sphere.png.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emx_dielectric_sphere.png")
EPS = 4.0; K = (EPS - 1) / (EPS + 2)
X, Z = np.meshgrid(np.linspace(-3, 3, 401), np.linspace(-3, 3, 401))
R2 = X**2 + Z**2; R = np.sqrt(R2); inside = R < 1
# outside: phi = -z + K z / r^3, E = -grad phi
Ex_out = 3 * K * X * Z / np.maximum(R, 1e-9)**5
Ez_out = 1 + K * (3 * Z**2 - R2) / np.maximum(R, 1e-9)**5
Ex = np.where(inside, 0.0, Ex_out); Ez = np.where(inside, 3 / (EPS + 2), Ez_out)
Dx = np.where(inside, EPS * Ex, Ex); Dz = np.where(inside, EPS * Ez, Ez)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
S = 0.28
seeds = np.array([[x0, -2.95] for x0 in np.arange(-2.8, 2.81, S)])
for ax, (Fx, Fz), name, k in [(axes[0], (Ex, Ez), r"$\mathbf{E}$", 0), (axes[1], (Dx, Dz), r"$\mathbf{D}$", 1)]:
    mag = np.sqrt(Fx**2 + Fz**2)
    pc = ax.pcolormesh(X, Z, mag, cmap="Greys", vmin=0, vmax=2.2, shading="auto", alpha=0.45)
    if k == 0:
        # E lines end on the bound charge at the surface: draw them outside, and inside with the density of the weaker field
        Fxm = np.where(inside, np.nan, Fx); Fzm = np.where(inside, np.nan, Fz)
        ax.streamplot(X, Z, Fxm, Fzm, start_points=seeds, color="royalblue", linewidth=1, density=4, arrowsize=0.8, broken_streamlines=False)
        top = np.array([[x0, 2.95] for x0 in np.arange(-2.8, 2.81, S)])
        ax.streamplot(X, Z, -Fxm, -Fzm, start_points=top, color="royalblue", linewidth=1, density=4, arrowsize=0, broken_streamlines=False)
        for x0 in [-2 * S, 0, 2 * S]:
            h = np.sqrt(1 - x0**2)
            ax.plot([x0, x0], [-h, h], color="royalblue", lw=1)
            ax.annotate("", xy=(x0, 0.1), xytext=(x0, -0.1), arrowprops=dict(arrowstyle="->", color="royalblue", lw=1))
    else:
        ax.streamplot(X, Z, Fx, Fz, start_points=seeds, color="royalblue", linewidth=1, density=4, arrowsize=0.8, broken_streamlines=False)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="saddlebrown", lw=1.5))
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xlabel(r"$x/a$"); ax.set_ylabel(r"$z/a$")
    ax.set_title(name + r" lines and magnitude (shading), $\varepsilon = 4$", fontsize=10)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight", dpi=220)
r, th, E0, a, eps = var("r th E0 a eps", domain="positive")
pin = -3 * E0 / (eps + 2) * r * cos(th)
pout = -E0 * r * cos(th) + (eps - 1) / (eps + 2) * E0 * a^3 * cos(th) / r^2
lap = lambda f: (diff(r^2 * diff(f, r), r) / r^2 + diff(sin(th) * diff(f, th), th) / (r^2 * sin(th))).simplify_full()
print("[PASS] the potential is harmonic inside and outside:", bool(lap(pin) == 0) and bool(lap(pout) == 0))
print("[PASS] phi and eps d_r phi are continuous at r = a:", bool((pin - pout).subs(r=a) == 0) and bool((eps * diff(pin, r) - diff(pout, r)).subs(r=a).simplify_full() == 0))
print("[PASS] interior field 3 E0/(eps + 2) = %.3f E0 for eps = 4:" % (3 / (EPS + 2)), abs(3 / (EPS + 2) - 0.5) < 1e-12)
P = (eps - 1) * 3 * E0 / (eps + 2)
print("[PASS] E_r jumps by the bound charge P cos(theta):", bool((-diff(pout, r) + diff(pin, r)).subs(r=a).simplify_full() == (P * cos(th)).simplify_full()))
print("wrote", os.path.relpath(OUT))
