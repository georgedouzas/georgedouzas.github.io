# Figure: power-law running (Dienes-Dudas-Gherghetta eq. 3.13): MSSM inverse couplings, logarithmic up to mu0 = 1e5 GeV,
# then with Kaluza-Klein towers, delta = 1, b~ = (3/5, -3, -6). Output: figures/ren-power-law.svg and an interactive version.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-power-law.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
MZ = 91.17; ainv = [98.29 * 3 / 5, 29.61, 8.5]; b = [33 / 5, 1, -3]; bt = [3 / 5, -3, -6]; mu0 = 1e5; d = 1
X = float(pi^(d / 2) / gamma(1 + d / 2))
def a(i, L):
    r = ainv[i] - b[i] / (2 * np.pi) * np.log(L / MZ)
    if L > mu0: r += bt[i] / (2 * np.pi) * np.log(L / mu0) - bt[i] * X / (2 * np.pi * d) * ((L / mu0)**d - 1)
    return r
L = np.logspace(np.log10(MZ), 6.6, 400)
fig, ax = plt.subplots(figsize=(7, 4.3)); cols = ["royalblue", "darkorange", "firebrick"]
for i in range(3):
    ax.semilogx(L, [a(i, x) for x in L], color=cols[i], lw=1.8, label=r"$\alpha_%d^{-1}$" % (i + 1))
ax.axvline(mu0, color="gray", ls=":", lw=1); ax.text(mu0 * 1.1, 5, r"$\mu_0 = 1/R$", fontsize=8)
ax.set_xlabel(r"$\mu$ or cutoff $\Lambda$ (GeV)"); ax.set_ylabel(r"$\alpha_i^{-1}$"); ax.set_ylim(0, 65); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
L12 = brentq(lambda x: a(0, x) - a(1, x), mu0 * 1.0001, mu0 * 1e3)
print("[PASS] alpha_1 = alpha_2 at Lambda = 20.3 mu0 with alpha^-1 = 45.05:", abs(L12 / mu0 - 20.34) < 0.05 and abs(a(0, L12) - 45.05) < 0.05)
print("[PASS] alpha_3 is within 6% there:", abs(a(2, L12) / a(0, L12) - 1) < 0.06)
print("[PASS] X_1 = 2, X_2 = pi, X_3 = 4 pi/3:", bool(pi^(1/2) / gamma(3 / 2) == 2) and bool(pi / gamma(2) == pi) and bool(pi^(3/2) / gamma(5 / 2) == 4 * pi / 3))
print("wrote", os.path.relpath(OUT))
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
tr = [{"x": list(L), "y": [a(i, x) for x in L], "mode": "lines", "name": "1/alpha_%d" % (i + 1), "line": {"color": cols[i], "width": 2}} for i in range(3)]
plotly_html("ren-power-law", tr, {"xaxis": {"title": "mu or cutoff (GeV)", "type": "log"}, "yaxis": {"title": "1/alpha_i", "range": [0, 65]}})
print("wrote figures/interactive/ren-power-law.html")
