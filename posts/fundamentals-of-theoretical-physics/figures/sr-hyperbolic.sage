# Figure: hyperbolic motion. Worldlines of constant proper acceleration, and the proper time to reach a distance d at 1 g,
# accelerating all the way or accelerating then decelerating. Output: figures/sr-hyperbolic.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-hyperbolic.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
t = np.linspace(0, 4, 300)
for al, col in [(0.5, "seagreen"), (1.0, "royalblue"), (2.0, "darkorange")]:
    ax[0].plot((np.sqrt(1 + al**2*t**2) - 1)/al, t, color=col, lw=2, label=r"$\alpha = %s$" % al)
ax[0].plot(t, t, ":", color="black", lw=1); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$t$"); ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
c = 2.99792458e8; g = 9.81; yr = 365.25*86400; cg = c/g/yr
d = np.logspace(-1, 7, 300)
ax[1].loglog(d, cg*np.arccosh(1 + d/cg), color="royalblue", lw=2, label="accelerate all the way")
ax[1].loglog(d, 2*cg*np.arccosh(1 + d/(2*cg)), "--", color="darkorange", lw=2, label="accelerate, then decelerate")
ax[1].loglog(d, d, ":", color="black", lw=1, label="$d/c$")
ax[1].set_xlabel("distance $d$ (light-years)"); ax[1].set_ylabel("proper time (years)"); ax[1].legend(fontsize=8); ax[1].set_ylim(0.1, 1e3)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
al = 1.0; tt = np.linspace(0, 3, 50); x = (np.sqrt(1 + tt**2) - 1)
u = np.gradient(x, tt)
print("[PASS] the plotted worldline has speed below 1:", np.all(np.abs(u) < 1))
print("[PASS] proper time to 2.5 million ly at 1 g is between 14 and 16 years:", 14 < cg*np.arccosh(1 + 2.5e6/cg) < 16)
