# Figure: three renormalisation group maps iterated as cobwebs: 1D Ising decimation K' = log(cosh 2K)/2, the approximate
# 2D Ising map K' = (3/8) log cosh 4K, and triangular-lattice percolation p' = p^3 + 3p^2(1 - p).
# Output: figures/ren-block-spin.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-block-spin.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
maps = [(lambda K: 0.5 * np.log(np.cosh(2 * K)), (0, 2), [1.8], "1D Ising: $K' = \\frac{1}{2}\\log\\cosh 2K$"),
        (lambda K: 3 / 8 * np.log(np.cosh(4 * K)), (0, 0.9), [0.49, 0.525], "2D Ising (approx.): $K' = \\frac{3}{8}\\log\\cosh 4K$"),
        (lambda p: p**3 + 3 * p**2 * (1 - p), (0, 1), [0.45, 0.55], "percolation: $p' = p^3 + 3p^2(1 - p)$")]
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
for a, (M, (lo, hi), starts, title) in zip(ax, maps):
    x = np.linspace(lo, hi, 300)
    a.plot(x, M(x), color="black", lw=1.6); a.plot(x, x, color="gray", lw=0.8, ls="--")
    for x0, c in zip(starts, ["royalblue", "firebrick"]):
        xs = [x0]
        for _ in range(12): xs.append(M(xs[-1]))
        px, py = [xs[0]], [xs[0]]
        for i in range(len(xs) - 1):
            px += [xs[i], xs[i + 1]]; py += [xs[i + 1], xs[i + 1]]
        a.plot(px, py, color=c, lw=0.9)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_title(title, fontsize=9); a.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT)
Ks = brentq(lambda K: 3 / 8 * np.log(np.cosh(4 * K)) - K, 0.3, 0.8)
print("[PASS] the approximate 2D fixed point is K* = 0.507:", abs(Ks - 0.5070) < 5e-4)
print("[PASS] the 1D map has K' < K for K in (0, 3]:", all(0.5 * np.log(np.cosh(2 * K)) < K for K in np.linspace(0.01, 3, 300)))
print("[PASS] percolation: p* = 1/2 is fixed and M'(1/2) = 3/2:", bool(QQ(1/2)^3 + 3 * QQ(1/2)^2 * QQ(1/2) == QQ(1/2)) and bool(diff(var('p')^3 + 3 * var('p')^2 * (1 - var('p')), var('p')).subs(p=1/2) == 3 / 2))
print("wrote", os.path.relpath(OUT))
