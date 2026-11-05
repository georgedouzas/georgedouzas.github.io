# Figure: angular distribution of the power radiated by a charge with acceleration parallel to its velocity,
# sin^2(alpha)/(1 - v cos(alpha))^5, normalised to its maximum, for v = 0, 0.5, 0.9 (polar diagram). Checks: v = 0 gives the Larmor
# pattern with total q^2 a^2/(6 pi); the maximum lies at cos(alpha) = (sqrt(1 + 15 v^2) - 1)/(3 v); the angular integral of the
# emitted power gives Lienard's gamma^6 factor. Output: figures/emw_larmor.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emw_larmor.svg")
al = np.linspace(0, 2 * np.pi, 1441)
fig = plt.figure(figsize=(6.5, 5.5)); ax = fig.add_subplot(1, 1, 1, projection="polar")
for v, col in [(0.0, "grey"), (0.5, "royalblue"), (0.9, "firebrick")]:
    f = np.sin(al)**2 / (1 - v * np.cos(al))**5
    ax.plot(al, f / f.max(), color=col, lw=2, label=r"$v = %.1f$" % v)
ax.set_title(r"$dP/d\Omega$ for $\mathbf{a} \parallel \mathbf{v}$ (velocity along $0^\circ$)", fontsize=10); ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
a_, q, v, x = var("a_ q v x"); th = var("th")
print("[PASS] v = 0: the Larmor pattern integrates to q^2 a^2/(6 pi):", bool(integrate(q^2 * a_^2 * sin(th)^2 / (16 * pi^2) * 2 * pi * sin(th), th, 0, pi) == q^2 * a_^2 / (6 * pi)))
vals = []
for vv in [0.5, 0.9]:
    aa = np.linspace(0, np.pi, 200001); f = np.sin(aa)**2 / (1 - vv * np.cos(aa))**5
    vals.append(abs(np.cos(aa[np.argmax(f)]) - (np.sqrt(1 + 15 * vv**2) - 1) / (3 * vv)) < 1e-4)
print("[PASS] the maximum lies at cos(alpha) = (sqrt(1 + 15 v^2) - 1)/(3 v) for v = 0.5, 0.9:", all(vals))
# power at emission time: integral of sin^2/(1 - v cos)^5 over the sphere equals (8 pi/3) gamma^6 for a || v
vv = 0.7; g = 1 / np.sqrt(1 - vv**2); aa = np.linspace(0, np.pi, 200001)
I_ = np.trapezoid(np.sin(aa)**2 / (1 - vv * np.cos(aa))**5 * 2 * np.pi * np.sin(aa), aa)
print("[PASS] the angular integral is (8 pi/3) gamma^6, Lienard's formula for a || v (v = 0.7): %.6f vs %.6f" % (I_, 8 * np.pi / 3 * g**6), abs(I_ / (8 * np.pi / 3 * g**6) - 1) < 1e-6)
print("wrote", os.path.relpath(OUT))
