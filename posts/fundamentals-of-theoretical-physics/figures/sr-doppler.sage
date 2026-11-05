# Figure: the relativistic Doppler factor 1/(gamma (1 - v cos theta)) against the angle for three speeds.
# Output: figures/sr-doppler.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-doppler.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
th = np.linspace(0, np.pi, 400)
fig, ax = plt.subplots(figsize=(6, 4.5))
for v, col in [(0.1, "seagreen"), (0.5, "royalblue"), (0.9, "darkorange")]:
    g = 1/np.sqrt(1 - v*v); ax.plot(np.degrees(th), 1/(g*(1 - v*np.cos(th))), color=col, lw=2, label="$v = %s$" % v)
ax.axvline(90, color="black", ls=":", lw=1); ax.axhline(1, color="black", lw=0.5)
ax.set_yscale("log"); ax.set_xlabel(r"angle $\theta$ between velocity and light (degrees)"); ax.set_ylabel(r"$\omega/\omega_0$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
v = 0.9; g = 1/np.sqrt(1 - v*v)
print("[PASS] forward factor equals sqrt((1+v)/(1-v)) at v = 0.9:", abs(1/(g*(1 - v)) - np.sqrt(1.9/0.1)) < 1e-12)
print("[PASS] transverse factor equals 1/gamma:", abs(1/(g*(1 - v*np.cos(np.pi/2))) - 1/g) < 1e-12)
