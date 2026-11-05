# Figure: one-loop running of the three inverse gauge couplings from M_Z (alpha_1 = 0.016985, alpha_2 = 0.03358, alpha_3 =
# 0.118, Nath eq. 9.33) with Standard Model and MSSM coefficients. Output: figures/ren-unification.svg and an
# interactive version figures/interactive/ren-unification.html.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-unification.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
MZ = 91.1876; ainv = [1 / 0.016985, 1 / 0.03358, 1 / 0.118]
bSM = [41 / 10, -19 / 6, -7]; bMS = [33 / 5, 1, -3]
mu = np.logspace(np.log10(MZ), 19, 300); t = np.log(mu / MZ)
fig, ax = plt.subplots(figsize=(7, 4.3))
cols = ["royalblue", "darkorange", "firebrick"]
for i in range(3):
    ax.semilogx(mu, ainv[i] - bSM[i] / (2 * np.pi) * t, "--", color=cols[i], lw=1.3)
    ax.semilogx(mu, ainv[i] - bMS[i] / (2 * np.pi) * t, color=cols[i], lw=1.8, label=r"$\alpha_%d^{-1}$" % (i + 1))
ax.set_xlabel(r"$\mu$ (GeV)"); ax.set_ylabel(r"$\alpha_i^{-1}(\mu)$"); ax.set_ylim(0, 65); ax.legend(fontsize=8, title="solid: MSSM, dashed: SM", title_fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
def cross(b, i, j):
    tt = (ainv[i] - ainv[j]) / ((b[i] - b[j]) / (2 * np.pi)); return MZ * np.exp(tt), ainv[i] - b[i] / (2 * np.pi) * tt
cs = [cross(bMS, i, j) for (i, j) in [(0, 1), (1, 2), (0, 2)]]
print("[PASS] the MSSM crossings lie between 1e16 and 4e16 GeV at alpha^-1 = 24.2-24.6:", all(1e16 < c[0] < 4e16 and 24.1 < c[1] < 24.7 for c in cs))
print("[PASS] the SM crossings of (1,2) and (2,3) differ by more than four decades:", np.log10(cross(bSM, 1, 2)[0] / cross(bSM, 0, 1)[0]) > 4)
print("[PASS] the b_i are (41/10, -19/6, -7) from (0, -22/3, -11) + 3 (4/3) + (1/10, 1/6, 0):", all(abs(x - y) < 1e-12 for x, y in zip(bSM, [0 + 4 + 0.1, -22 / 3 + 4 + 1 / 6, -11 + 4])))
print("wrote", os.path.relpath(OUT))
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
tr = []
for i in range(3):
    tr.append({"x": list(mu), "y": list(ainv[i] - bMS[i] / (2 * np.pi) * t), "mode": "lines", "name": "MSSM 1/alpha_%d" % (i + 1), "line": {"color": cols[i], "width": 2.5}})
    tr.append({"x": list(mu), "y": list(ainv[i] - bSM[i] / (2 * np.pi) * t), "mode": "lines", "name": "SM 1/alpha_%d" % (i + 1), "line": {"color": cols[i], "width": 1.5, "dash": "dash"}})
plotly_html("ren-unification", tr, {"xaxis": {"title": "mu (GeV)", "type": "log"}, "yaxis": {"title": "1/alpha_i", "range": [0, 65]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/ren-unification.html")
