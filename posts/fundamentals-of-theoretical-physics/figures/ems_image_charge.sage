# Figure: a charge q at height a = 1 above a grounded conducting plane (Heaviside-Lorentz units, q = 1). Left, field lines
# and equipotentials in z > 0 (the field of q and its image -q at z = -1). Right, the induced surface charge sigma(s) a^2/q
# and the fraction of the induced charge within radius s. Checks: phi = 0 on the plane; the total induced charge is -q;
# half of it lies within s = sqrt(3) a; the field just above the plane is normal with E_z = sigma.
# Output: figures/ems_image_charge.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ems_image_charge.svg")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
x = np.linspace(-3, 3, 401); z = np.linspace(0.0005, 3, 300); X, Z = np.meshgrid(x, z)
d1 = np.hypot(X, Z - 1); d2 = np.hypot(X, Z + 1)
Ex = (X / d1**3 - X / d2**3) / (4 * np.pi); Ez = ((Z - 1) / d1**3 - (Z + 1) / d2**3) / (4 * np.pi)
phi = (1 / d1 - 1 / d2) / (4 * np.pi)
ax.streamplot(X, Z, Ex, Ez, color="grey", density=1.4, linewidth=0.7, arrowsize=0.8)
ax.contour(X, Z, phi, levels=[0.005, 0.01, 0.02, 0.04, 0.08, 0.16], cmap="Reds", linewidths=1.0)
ax.fill_between([-3, 3], -0.4, 0, color="lightgray"); ax.plot([0], [1], "o", color="firebrick", ms=8)
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-0.4, 3); ax.set_xlabel(r"$x/a$"); ax.set_ylabel(r"$z/a$")
ax.set_title("charge above a grounded plane")
ax = axes[1]
s = np.linspace(0, 5, 500)
sigma = -1 / (2 * np.pi * (s**2 + 1)**1.5); frac = 1 - 1 / np.sqrt(s**2 + 1)
ax.plot(s, sigma, color="royalblue", lw=2, label=r"$\sigma a^2/q$")
ax2 = ax.twinx(); ax2.plot(s, frac, color="firebrick", lw=1.5, ls="--", label="fraction within $s$")
ax2.axhline(0.5, color="lightgray", lw=0.8); ax2.axvline(np.sqrt(3), color="lightgray", lw=0.8)
ax.set_xlabel(r"$s/a$"); ax.set_ylabel(r"$\sigma a^2/q$"); ax2.set_ylabel("fraction of the induced charge")
ax.legend(loc="center right"); ax2.legend(loc="lower right"); ax.set_title("induced surface charge")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
q, a, ss, xx, yy, zz = var('q a ss xx yy zz'); assume(a > 0, ss > 0)
Phi = q / (4 * pi * sqrt(xx^2 + yy^2 + (zz - a)^2)) - q / (4 * pi * sqrt(xx^2 + yy^2 + (zz + a)^2))
print("[PASS] phi = 0 on the plane z = 0:", bool(Phi.subs(zz=0) == 0))
sig = (-diff(Phi, zz)).subs(zz=0).simplify_full()
print("[PASS] sigma = E_z just above the plane = -q a/(2 pi (s^2 + a^2)^(3/2)):", bool((sig - (-q * a / (2 * pi * (xx^2 + yy^2 + a^2)^(3/2)))).simplify_full() == 0))
tot = integrate(2 * pi * ss * (-q * a / (2 * pi * (ss^2 + a^2)^(3/2))), ss, 0, oo)
print("[PASS] total induced charge -q:", bool(tot == -q))
half = integrate(2 * pi * ss * (-a / (2 * pi * (ss^2 + a^2)^(3/2))), ss, 0, sqrt(3) * a)
print("[PASS] half of the induced charge lies within s = sqrt(3) a:", bool(half == -1 / 2))
print("wrote", os.path.relpath(OUT))
