# Figure: the twin paradox. The stay-at-home and the traveller (v = 0.8, three years out and three back), with yearly
# marks of proper time and the Earth's yearly light signals received by the traveller.
# Output: figures/sr-twin.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-twin.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
v = 0.8; g = 1/np.sqrt(1 - v*v); Tt = 3*g; X = v*Tt
fig, ax = plt.subplots(figsize=(5, 7))
ax.plot([0, 0], [0, 2*Tt], color="black", lw=2)
ax.plot([0, X, 0], [0, Tt, 2*Tt], color="royalblue", lw=2)
for k in range(1, 11): ax.plot(0, k, "o", color="black", ms=4)
for k in range(1, 7):
    t = k*g; x = v*t if t <= Tt else X - v*(t - Tt)
    ax.plot(x, t, "o", color="royalblue", ms=4)
for k in range(1, 10):
    t_out = k/(1 - v)
    t = t_out if t_out <= Tt else (k + X + v*Tt)/(1 + v)
    x = v*t if t <= Tt else X - v*(t - Tt)
    ax.plot([0, x], [k, t], color="darkorange", lw=0.8)
ax.set_xlabel("$x$ (light-years)"); ax.set_ylabel("$t$ (years)"); ax.set_xlim(-0.5, 5); ax.set_ylim(0, 10.5); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the traveller's proper time is 6 years:", abs(2*Tt/g - 6) < 1e-12)
print("[PASS] 10 years pass at home:", abs(2*Tt - 10) < 1e-12)
