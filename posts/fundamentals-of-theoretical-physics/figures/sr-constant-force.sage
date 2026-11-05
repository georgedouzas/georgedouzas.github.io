# Figure: motion under a constant three-force (f/m = 1) from rest: velocity and position against time, relativistic
# against Newtonian. Output: figures/sr-constant-force.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-constant-force.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 5, 300)
v = t/np.sqrt(1 + t**2); x = np.sqrt(1 + t**2) - 1
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(t, v, color="royalblue", lw=2, label="relativistic"); ax[0].plot(t[t < 1.6], t[t < 1.6], "k--", lw=1, label="Newtonian $v = ft/m$")
ax[0].axhline(1, color="darkorange", ls=":", label="speed of light"); ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$v$"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.6)
ax[1].plot(t, x, color="royalblue", lw=2, label="relativistic"); ax[1].plot(t, t**2/2, "k--", lw=1, label="Newtonian")
ax[1].plot(t, t - 1, color="darkorange", ls=":", label="asymptote $x = t - 1$"); ax[1].set_xlabel("$t$"); ax[1].set_ylabel("$x$"); ax[1].set_ylim(0, 6); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
p = v/np.sqrt(1 - v**2)
print("[PASS] the relativistic momentum grows linearly, p = f t:", np.allclose(p, t))
