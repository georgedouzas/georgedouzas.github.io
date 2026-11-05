# Figure: field lines and equipotentials of two point charges at x = -1, +1 in a plane through them (Heaviside-Lorentz
# units, q = 1): left opposite charges +q, -q, right equal charges +q, +q. Checks: the field of the equal pair vanishes at
# the midpoint; the flux through a small sphere around a charge is q (Gauss's law); the potential of
# the opposite pair vanishes on the symmetry plane; field lines of the equal pair keep cos(theta1) + cos(theta2) fixed.
# Output: figures/ems_coulomb_fields.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ems_coulomb_fields.svg")

def field(charges, X, Y):
    Ex = np.zeros_like(X); Ey = np.zeros_like(X); phi = np.zeros_like(X)
    for (q, cx, cy) in charges:
        dx, dy = X - cx, Y - cy; r = np.sqrt(dx**2 + dy**2)
        Ex += q * dx / (4 * np.pi * r**3); Ey += q * dy / (4 * np.pi * r**3); phi += q / (4 * np.pi * r)
    return Ex, Ey, phi

x = np.linspace(-3, 3, 401); X, Y = np.meshgrid(x, x)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, charges, title in [(axes[0], [(1, -1, 0), (-1, 1, 0)], r"$+q$ and $-q$"), (axes[1], [(1, -1, 0), (1, 1, 0)], r"$+q$ and $+q$")]:
    Ex, Ey, phi = field(charges, X, Y)
    ax.streamplot(X, Y, Ex, Ey, color="grey", density=1.4, linewidth=0.7, arrowsize=0.8)
    levels = np.sort(np.concatenate([np.linspace(-0.3, 0.3, 13)]))
    ax.contour(X, Y, phi, levels=levels, cmap="coolwarm", linewidths=1.0)
    for (q, cx, cy) in charges:
        ax.plot(cx, cy, "o", color="firebrick" if q > 0 else "royalblue", markersize=9)
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_title(title); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$z$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")

# checks
Ex, Ey, _ = field([(1, -1, 0), (1, 1, 0)], np.array([0.0]), np.array([0.0]))
print("[PASS] equal charges: the field vanishes at the midpoint:", abs(Ex[0]) < 1e-14 and abs(Ey[0]) < 1e-14)
# flux through a sphere of radius 0.5 around the charge at (-1, 0, 0), by quadrature in 3D (axial symmetry about the x axis)
th = np.linspace(0, np.pi, 2001); rs = 0.5
px = -1 + rs * np.cos(th); ps = rs * np.sin(th)
def E3(qs, px, ps):
    Ex = 0 * px; Es = 0 * px
    for (q, cx) in qs:
        d = np.sqrt((px - cx)**2 + ps**2); Ex += q * (px - cx) / (4 * np.pi * d**3); Es += q * ps / (4 * np.pi * d**3)
    return Ex, Es
for qs, name in [([(1, -1), (-1, 1)], "opposite"), ([(1, -1), (1, 1)], "equal")]:
    Exs, Ess = E3(qs, px, ps)
    En = Exs * np.cos(th) + Ess * np.sin(th)
    flux = np.trapezoid(En * 2 * np.pi * rs**2 * np.sin(th), th)
    print("[PASS] %s pair: flux through a sphere of radius 1/2 around +q is q = 1 (Gauss's law): %.6f" % (name, flux), abs(flux - 1) < 1e-5)
_, _, phi0 = field([(1, -1, 0), (-1, 1, 0)], np.array([0.0, 0.0]), np.array([0.3, 2.0]))
print("[PASS] opposite pair: the potential vanishes on the symmetry plane:", np.allclose(phi0, 0))
# integrate one field line of the equal pair and check that cos(theta1) + cos(theta2) is constant along it
from scipy.integrate import solve_ivp
def rhs(t, p):
    Ex, Es = E3([(1, -1), (1, 1)], np.array([p[0]]), np.array([p[1]])); n = np.hypot(Ex[0], Es[0]); return [Ex[0] / n, Es[0] / n]
sol = solve_ivp(rhs, [0, 3], [-1 + 0.05 * np.cos(1.0), 0.05 * np.sin(1.0)], rtol=1e-10, atol=1e-12)
c = [((p0 + 1) / np.hypot(p0 + 1, p1)) + ((p0 - 1) / np.hypot(p0 - 1, p1)) for p0, p1 in zip(sol.y[0], sol.y[1])]
print("[PASS] equal pair: cos(theta1) + cos(theta2) is constant along a field line (spread %.1e):" % (max(c) - min(c)), max(c) - min(c) < 1e-6)
print("wrote", os.path.relpath(OUT))
