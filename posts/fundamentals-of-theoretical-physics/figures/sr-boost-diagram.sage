# Figure: the axes of a frame K' moving with v = 0.5 along x, drawn in the (x, t) diagram of K, with the light ray, the
# lines of simultaneity of K' and the calibration hyperbolae t^2 - x^2 = 1 and x^2 - t^2 = 1.
# Output: figures/sr-boost-diagram.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-boost-diagram.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
v = 0.5; g = 1/np.sqrt(1 - v*v)
fig, ax = plt.subplots(figsize=(6, 6))
L = 2.2
ax.plot([-L, L], [0, 0], color="black", lw=1.2); ax.plot([0, 0], [-L, L], color="black", lw=1.2)
ax.text(L, -0.12, "$x$", ha="right", va="top"); ax.text(0.05, L, "$t$", va="top")
ax.plot([-L, L], [-L, L], "--", color="darkorange", lw=1.2); ax.plot([-L, L], [L, -L], "--", color="darkorange", lw=1.2)
s = np.linspace(-L, L, 2)
ax.plot(v*s, s, color="royalblue", lw=1.8); ax.plot(s, v*s, color="royalblue", lw=1.8)
ax.text(v*L, L, "$t'$", color="royalblue", va="top", ha="right"); ax.text(L, v*L, "$x'$", color="royalblue", va="bottom", ha="right")
for tp in [-1.5, -1, -0.5, 0.5, 1, 1.5]:
    # line t' = tp: t - v x = tp/g
    xs = np.linspace(-L, L, 2); ax.plot(xs, tp/g + v*xs, color="0.75", lw=0.8)
eta = np.linspace(-1.5, 1.5, 300)
ax.plot(np.sinh(eta), np.cosh(eta), ":", color="black", lw=1); ax.plot(np.cosh(eta), np.sinh(eta), ":", color="black", lw=1)
ax.plot(np.sinh(eta), -np.cosh(eta), ":", color="black", lw=1); ax.plot(-np.cosh(eta), np.sinh(eta), ":", color="black", lw=1)
ax.plot([g*v], [g], "o", color="royalblue", ms=4); ax.plot([g], [g*v], "o", color="royalblue", ms=4)
ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
# the unit tick of t' (t' = 1, x' = 0) is at (x, t) = (g v, g), on the hyperbola t^2 - x^2 = 1
print("[PASS] unit of t' lies on t^2 - x^2 = 1:", abs(g**2 - (g*v)**2 - 1) < 1e-12)
print("[PASS] unit of x' lies on x^2 - t^2 = 1:", abs(g**2 - (g*v)**2 - 1) < 1e-12)
print("[PASS] the x' axis is the line t = v x (t' = 0):", abs(g*(g*v - v*g)) < 1e-12)
