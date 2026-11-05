# Figure: the one-loop flow dr/dl = 2r + (n+2)/3 g/(1+r), dg/dl = eps g - (n+8)/3 g^2/(1+r)^2 for n = 1, eps = 1, towards the
# infrared, with the Gaussian and Wilson-Fisher fixed points. Output: figures/ren-wilson-fisher.svg and an
# interactive version.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-wilson-fisher.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
n, eps = 1, 1.0
F = lambda r, g: (2 * r + (n + 2) / 3 * g / (1 + r), eps * g - (n + 8) / 3 * g**2 / (1 + r)**2)
rs, gs = fsolve(lambda v: F(*v), [-0.15, 0.3])
R, Gg = np.meshgrid(np.linspace(-0.45, 0.4, 40), np.linspace(0, 0.6, 40))
U, V = F(R, Gg)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.streamplot(R, Gg, U, V, density=1.2, color="lightsteelblue", linewidth=0.8)
# critical surface: integrate backwards from near the fixed point along the stable eigenvector
J = np.array([[2 - (n + 2) / 3 * gs / (1 + rs)**2, (n + 2) / 3 / (1 + rs)], [2 * (n + 8) / 3 * gs**2 / (1 + rs)**3, eps - 2 * (n + 8) / 3 * gs / (1 + rs)**2]])
w, vecs = np.linalg.eig(J); k = np.argmin(w); v = vecs[:, k]
for sgn in [1, -1]:
    sol = solve_ivp(lambda t, y: [-c for c in F(*y)], [0, 8], [rs + 1e-3 * sgn * v[0], gs + 1e-3 * sgn * v[1]], max_step=0.01)
    ax.plot(sol.y[0], sol.y[1], color="black", lw=1.4)
ax.plot(0, 0, "ko", ms=6); ax.plot(rs, gs, "o", color="firebrick", ms=7)
ax.text(0.01, 0.015, "Gaussian", fontsize=8); ax.text(rs + 0.02, gs + 0.02, "Wilson-Fisher", fontsize=8, color="firebrick")
ax.set_xlim(-0.45, 0.4); ax.set_ylim(0, 0.6); ax.set_xlabel("$r$"); ax.set_ylabel("$g$")
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the Wilson-Fisher point of the truncated flow at n = 1, eps = 1 is (r*, g*) = (-1/7, 12/49):", abs(rs + 1 / 7) < 1e-6 and abs(gs - 12 / 49) < 1e-6)
print("[PASS] it has one positive and one negative eigenvalue:", sorted(np.sign(w)) == [-1, 1])
print("[PASS] the first-order values -eps/6, eps/3 are within 0.1 of it:", abs(rs + 1 / 6) < 0.1 and abs(gs - 1 / 3) < 0.1)
print("wrote", os.path.relpath(OUT))
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
tr = []
for r0 in np.linspace(-0.4, 0.35, 10):
    for g0 in [0.05, 0.55]:
        sol = solve_ivp(lambda t, y: list(F(*y)), [0, 3], [r0, g0], max_step=0.02, events=lambda t, y: abs(y[0]) - 1.0)
        tr.append({"x": list(sol.y[0]), "y": list(sol.y[1]), "mode": "lines", "line": {"color": "steelblue", "width": 1}, "showlegend": False})
tr.append({"x": [0, rs], "y": [0, gs], "mode": "markers", "marker": {"size": 10, "color": ["black", "firebrick"]}, "name": "fixed points"})
plotly_html("ren-wilson-fisher", tr, {"xaxis": {"title": "r", "range": [-0.5, 0.45]}, "yaxis": {"title": "g", "range": [0, 0.65]}})
print("wrote figures/interactive/ren-wilson-fisher.html")
