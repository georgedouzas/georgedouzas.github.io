# Figure: Ising critical exponents: mean field, the epsilon expansion at first and second order at eps = 1 and 2 (other
# exponents from nu and eta via the scaling relations), and reference values (3D numerical, 2D exact).
# Output: figures/ren-exponents.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-exponents.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
def expo(e, order, n=1):
    nu = 0.5 + (n + 2) * e / (4 * (n + 8)) + (order >= 2) * (n + 2) * (n**2 + 23 * n + 60) * e**2 / (8 * (n + 8)**3)
    eta = float((order >= 2) * (n + 2) * e**2 / (2 * (n + 8)**2)); d = float(4 - e); nu = float(nu)
    return {"nu": nu, "eta": eta, "gamma": nu * (2 - eta), "alpha": 2 - d * nu, "beta": nu * (d - 2 + eta) / 2, "delta": (d + 2 - eta) / (d - 2 + eta) if d - 2 + eta != 0 else float("inf")}
names = ["alpha", "beta", "gamma", "delta", "nu", "eta"]
mf = {"alpha": 0, "beta": 0.5, "gamma": 1, "delta": 3, "nu": 0.5, "eta": 0}
ref3 = {"alpha": 0.110, "beta": 0.326, "gamma": 1.237, "delta": 4.79, "nu": 0.630, "eta": 0.036}
ref2 = {"alpha": 0, "beta": 1 / 8, "gamma": 7 / 4, "delta": 15, "nu": 1, "eta": 1 / 4}
fig, axs = plt.subplots(1, 6, figsize=(12, 3.4))
for a, nm in zip(axs, names):
    a.plot([0], [mf[nm]], "s", color="gray", ms=6)
    for x, e, col, ref in [(1, 1, "royalblue", ref3), (2, 2, "firebrick", ref2)]:
        a.plot([x - 0.12], [expo(e, 1)[nm]], "o", mfc="none", color=col, ms=6)
        v2 = expo(e, 2)[nm]
        if v2 < 20: a.plot([x], [v2], "o", color=col, ms=6)
        a.plot([x + 0.14], [ref[nm]], "kx", ms=7)
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["MF", "d=3", "d=2"], fontsize=8); a.set_title(r"$\%s$" % nm, fontsize=11); a.set_xlim(-0.5, 2.5)
axs[3].set_ylim(0, 20)
fig.tight_layout(); fig.savefig(OUT)
e3, e2 = expo(1, 2), expo(2, 2)
print("[PASS] second order at eps = 1: nu = 0.627, eta = 0.019, gamma = 1.241:", abs(e3["nu"] - 0.6265) < 1e-3 and abs(e3["eta"] - 1 / 54) < 1e-9 and abs(e3["gamma"] - 1.2415) < 1e-3)
print("[PASS] second order at eps = 2: nu = 0.840, eta = 2/27, delta = 53:", abs(e2["nu"] - 0.8395) < 1e-3 and abs(e2["eta"] - 2 / 27) < 1e-12 and abs(e2["delta"] - 53) < 1e-9)
print("[PASS] the exact 2D values satisfy gamma = nu (2 - eta) and delta = (d + 2 - eta)/(d - 2 + eta):", abs(ref2["gamma"] - ref2["nu"] * (2 - ref2["eta"])) < 1e-12 and abs(ref2["delta"] - (4 - ref2["eta"]) / ref2["eta"]) < 1e-12)
print("wrote", os.path.relpath(OUT))
