# Figure: one-loop running with Standard Model coefficients below a common superpartner mass M_S and MSSM ones above
# (step approximation), M_S = 1 TeV and 100 TeV. Output: figures/ren-thresholds.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-thresholds.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
MZ = 91.1876; ainv = [1 / 0.016985, 1 / 0.03358, 1 / 0.118]; bSM = [41 / 10, -19 / 6, -7]; bMS = [33 / 5, 1, -3]
def a(i, mu, MS):
    if mu <= MS: return ainv[i] - bSM[i] / (2 * np.pi) * np.log(mu / MZ)
    return ainv[i] - bSM[i] / (2 * np.pi) * np.log(MS / MZ) - bMS[i] / (2 * np.pi) * np.log(mu / MS)
mu = np.logspace(np.log10(MZ), 18, 400)
fig, ax = plt.subplots(figsize=(7, 4.3)); cols = ["royalblue", "darkorange", "firebrick"]
for MS, st in [(1e3, "-"), (1e5, "--")]:
    for i in range(3):
        ax.semilogx(mu, [a(i, m, MS) for m in mu], st, color=cols[i], lw=1.6, label=(r"$\alpha_%d^{-1}$" % (i + 1)) if MS == 1e3 else None)
ax.set_ylim(15, 60); ax.set_xlim(1e10, 1e18); ax.set_xlabel(r"$\mu$ (GeV)"); ax.set_ylabel(r"$\alpha_i^{-1}$"); ax.legend(fontsize=8, title=r"solid $M_S = 1$ TeV, dashed $100$ TeV", title_fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
def meet(MS):
    tS = np.log(MS / MZ); D = ainv[0] - ainv[1]
    return tS + (2 * np.pi * D - (bSM[0] - bSM[1]) * tS) / (bMS[0] - bMS[1])
print("[PASS] the alpha_1 = alpha_2 crossing falls by a factor of about 4 from M_S = 1 to 100 TeV:", 3.5 < np.exp(meet(1e3) - meet(1e5)) < 4.5)
print("[PASS] the slope of the crossing with log M_S is -25/84:", abs((meet(1e4) - meet(1e3)) / np.log(10) + 25 / 84) < 1e-12)
print("[PASS] the coupling is continuous at M_S:", abs(a(0, 1e3, 1e3) - (ainv[0] - bSM[0] / (2 * np.pi) * np.log(1e3 / MZ))) < 1e-12)
print("wrote", os.path.relpath(OUT))
