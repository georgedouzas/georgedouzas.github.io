# Figure: the beta function of the quartic theory in units of 16 pi^2, 3u^2 at one loop and 3u^2 - 17u^3/3 at two loops,
# u = lambda/(16 pi^2), with the spurious two-loop zero at u = 9/17. Output: figures/ren-two-loop-beta.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-two-loop-beta.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
u = np.linspace(0, 0.7, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(u, 3 * u**2, "k--", lw=1.4, label="one loop $3u^2$")
ax.plot(u, 3 * u**2 - 17 / 3 * u**3, color="firebrick", lw=1.8, label=r"two loops $3u^2 - \frac{17}{3}u^3$")
ax.axvline(9 / 17, color="gray", ls=":", lw=1); ax.text(9 / 17 + 0.01, 0.5, r"$u = 9/17$", fontsize=8)
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel(r"$u = \lambda/16\pi^2$"); ax.set_ylabel(r"$\beta/16\pi^2$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
var('uu')
print("[PASS] the two-loop beta function vanishes at u = 9/17:", QQ(9/17) in [r.rhs() for r in solve(3 * uu - 17 / 3 * uu^2 == 0, uu)])
print("[PASS] at u = 9/17, lambda = 16 pi^2 9/17 = 83.6:", abs(float(16 * pi^2 * 9 / 17) - 83.6) < 0.05)
print("wrote", os.path.relpath(OUT))
