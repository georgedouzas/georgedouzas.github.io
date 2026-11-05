# Figure: Einstein synchronisation and radar coordinates. A light signal from the worldline x = 0 at t1, reflected at an
# event E on a second worldline, returns at t2; E gets the time (t1 + t2)/2 and the distance (t2 - t1)/2.
# Output: figures/sr-radar.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-radar.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t1, t2 = 1.0, 3.0
tE, xE = (t1 + t2)/2, (t2 - t1)/2
fig, ax = plt.subplots(figsize=(4.5, 5))
ax.plot([0, 0], [0, 4], color="black", lw=2); ax.text(-0.08, 3.9, "$A$", ha="right")
ax.plot([0.5, 1.5], [0, 4], color="royalblue", lw=2); ax.text(1.5, 3.9, "$B$", color="royalblue")
ax.plot([0, xE], [t1, tE], color="darkorange", lw=1.5); ax.plot([xE, 0], [tE, t2], color="darkorange", lw=1.5)
ax.plot([0, xE], [tE, tE], ":", color="black"); ax.plot(xE, tE, "o", color="royalblue")
ax.text(xE + 0.05, tE, "$E$", va="center"); ax.text(-0.08, t1, "$t_1$", ha="right", va="center"); ax.text(-0.08, t2, "$t_2$", ha="right", va="center")
ax.text(-0.08, tE, r"$\frac{t_1 + t_2}{2}$", ha="right", va="center")
ax.set_xlim(-0.8, 2); ax.set_ylim(0, 4.1); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$t$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] E lies on B's worldline x = 0.5 + t/4:", abs(xE - (0.5 + tE/4)) < 1e-12)
print("[PASS] outgoing and returning signals are light rays:", abs((tE - t1) - xE) < 1e-12 and abs((t2 - tE) - xE) < 1e-12)
