# Figure: the headlight effect. Fraction of an isotropic emission (in the source frame) found within the laboratory
# angle theta of the direction of motion, for several speeds. Output: figures/sr-beaming.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-beaming.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
th = np.linspace(0, np.pi, 400)
fig, ax = plt.subplots(figsize=(6, 4.5))
for v, col in [(0.0, "black"), (0.5, "seagreen"), (0.9, "royalblue"), (0.99, "darkorange")]:
    cs = (np.cos(th) - v)/(1 - v*np.cos(th))             # source-frame cosine of a ray seen at lab angle theta
    ax.plot(np.degrees(th), (1 - cs)/2, color=col, lw=2, label="$v = %s$" % v)
ax.axhline(0.5, color="black", ls=":", lw=1)
ax.set_xlabel(r"laboratory angle $\theta$ from the direction of motion (degrees)"); ax.set_ylabel("fraction of the light within $\\theta$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
v = 0.9; th12 = np.arccos(v)
print("[PASS] half the light lies within cos(theta) = v:", abs((1 - (np.cos(th12) - v)/(1 - v*np.cos(th12)))/2 - 0.5) < 1e-12)
