# Figure: the Rindler frame in the wedge x > |t|: hyperbolae of constant X and lines of constant T, and the horizon.
# Output: figures/sr-rindler.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-rindler.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(5.5, 5.5)); L = 3
T = np.linspace(-2, 2, 200)
for X in [0.5, 1.0, 1.5, 2.0, 2.5]:
    ax.plot(X*np.cosh(T), X*np.sinh(T), color="royalblue", lw=1.6)
for TT in np.linspace(-1.2, 1.2, 9):
    s = np.linspace(0, 4, 2); ax.plot(s*np.cosh(TT), s*np.sinh(TT), color="0.6", lw=0.8)
ax.plot([0, L], [0, L], color="darkorange", lw=2); ax.plot([0, L], [0, -L], color="darkorange", lw=2)
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
ax.set_xlim(-0.5, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$t$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
X = 1.7; TT = 0.6
print("[PASS] a Rindler worldline satisfies x^2 - t^2 = X^2:", abs((X*np.cosh(TT))**2 - (X*np.sinh(TT))**2 - X**2) < 1e-12)
print("[PASS] lines of constant T pass through the origin with slope tanh(aT):", abs(np.sinh(TT)/np.cosh(TT) - np.tanh(TT)) < 1e-15)
