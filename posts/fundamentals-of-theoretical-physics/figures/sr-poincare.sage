# Figure: translations and boosts do not commute. A square of events translated by a = (0, 1) and boosted by L(0.6) in
# the two orders. Output: figures/sr-poincare.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-poincare.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
eta = 0.6; L = np.array([[np.cosh(eta), np.sinh(eta)], [np.sinh(eta), np.cosh(eta)]]); a = np.array([0., 1.])   # (t, x)
sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)*0.6 + np.array([0.2, 0.2])
A = np.array([L @ (p + a) for p in sq])          # boost after translation: (L, 0)(1, a)
B = np.array([L @ p + a for p in sq])            # translation after boost: (1, a)(L, 0)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for P, col, lab in [(sq, "black", "events"), (A, "royalblue", "boost after translation"), (B, "darkorange", "translation after boost")]:
    ax.plot(P[:, 1], P[:, 0], color=col, lw=2, label=lab); ax.fill(P[:, 1], P[:, 0], color=col, alpha=0.12)
ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the two images differ by the translation L a - a:", np.allclose(A - B, L @ a - a))
