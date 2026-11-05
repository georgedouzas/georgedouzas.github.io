# Figure: left, field E(r) and potential phi(r) of a uniformly charged ball (R = 1, Q = 4 pi, Heaviside-Lorentz units);
# right, equipotentials and field lines of two opposite line charges through (-1, 0) and (1, 0). Checks: E and phi are
# continuous at r = R; phi(0) = 3/2 phi(R); Poisson's equation inside; the level s2/s1 = K is the Apollonius circle.
# Output: figures/ems_potential.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ems_potential.svg")
r = var('r', domain='positive')
E_in, E_out = r, 1 / r^2                         # Q/(4 pi) = 1, R = 1
phi_in, phi_out = (3 - r^2) / 2, 1 / r
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
rr = np.linspace(0.001, 4, 800)
E = np.where(rr < 1, rr, 1 / rr**2); P = np.where(rr < 1, (3 - rr**2) / 2, 1 / rr)
ax.plot(rr, E, color="royalblue", lw=2, label=r"$E(r)$"); ax.plot(rr, P, color="firebrick", lw=2, label=r"$\phi(r)$")
ax.axvline(1, color="lightgray", ls="--"); ax.set_xlabel(r"$r/R$"); ax.legend(); ax.set_ylim(0, 1.7); ax.set_title("uniformly charged ball")
ax = axes[1]
x = np.linspace(-3, 3, 401); X, Y = np.meshgrid(x, x)
s1 = np.hypot(X + 1, Y); s2 = np.hypot(X - 1, Y)
phi = np.log(s2 / s1) / (2 * np.pi)
Ex = (X + 1) / s1**2 - (X - 1) / s2**2; Ey = Y / s1**2 - Y / s2**2
ax.streamplot(X, Y, Ex, Ey, color="grey", density=1.3, linewidth=0.7, arrowsize=0.8)
ax.contour(X, Y, phi, levels=np.linspace(-0.3, 0.3, 13), cmap="coolwarm", linewidths=1.0)
ax.plot([-1], [0], "o", color="firebrick", ms=8); ax.plot([1], [0], "o", color="royalblue", ms=8)
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_title("two opposite line charges")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] ball: E and phi continuous at r = R, and E = -phi':", bool(E_in.subs(r=1) == E_out.subs(r=1)) and bool(phi_in.subs(r=1) == phi_out.subs(r=1)) and bool(-diff(phi_in, r) == E_in) and bool(-diff(phi_out, r) == E_out))
print("[PASS] ball: phi(0) = 3/2 phi(R):", bool(phi_in.subs(r=0) == 3 / 2 * phi_out.subs(r=1)))
print("[PASS] ball: Poisson's equation inside, (1/r^2)(r^2 phi')' = -rho = -3:", bool((diff(r^2 * diff(phi_in, r), r) / r^2).simplify_full() == -3))
xx, yy, K, a = var('xx yy K a')
circ = (xx + a * (K^2 + 1) / (K^2 - 1))^2 + yy^2 - (2 * a * K / (K^2 - 1))^2
lev = (xx - a)^2 + yy^2 - K^2 * ((xx + a)^2 + yy^2)
print("[PASS] the level s2/s1 = K is the circle of Apollonius:", bool((lev + (K^2 - 1) * circ).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
