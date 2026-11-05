# Figure: schemes compared. Left: lambda in MS-bar and in momentum subtraction (lambda_MOM = lambda - 3 lambda^2/(16 pi^2)),
# running with the one-loop beta function from lambda_MSbar = 3. Right: 1/alpha with the electron alone: on-shell
# constant, MS-bar running, and the exact one-loop effective charge. Output: figures/ren-schemes.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-schemes.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
b = 3 / (16 * np.pi**2)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
t = np.linspace(-8, 8, 300)
lms = 3 / (1 - b * 3 * t)
lmom0 = 3 - b * 9
lmom = lmom0 / (1 - b * lmom0 * t)
ax[0].plot(t, lms, color="black", lw=1.8, label=r"$\overline{\mathrm{MS}}$")
ax[0].plot(t, lmom, "--", color="royalblue", lw=1.8, label="momentum subtraction")
ax[0].set_xlabel(r"$\log(\mu/\mu_0)$"); ax[0].set_ylabel(r"$\lambda(\mu)$"); ax[0].legend(fontsize=8); ax[0].set_title("the quartic coupling in two schemes", fontsize=10)
al = 1 / 137.035999; me = 0.000510999
def inv_eff(Q):
    I = quad(lambda x: x * (1 - x) * np.log(1 + x * (1 - x) * Q**2 / me**2), 0, 1)[0]
    return (1 - 2 * al / np.pi * I) / al
Q = np.logspace(np.log10(me) - 1, np.log10(100), 200)
ax[1].semilogx(Q, [1 / al] * len(Q), color="gray", lw=1.4, label="on-shell")
ax[1].semilogx(Q[Q >= me], 1 / al - 2 / (3 * np.pi) * np.log(Q[Q >= me] / me), color="black", lw=1.8, label=r"$\overline{\mathrm{MS}}$, $1/\alpha(\mu)$")
ax[1].semilogx(Q, [inv_eff(q) for q in Q], "--", color="firebrick", lw=1.8, label=r"effective, $1/\alpha_{\mathrm{eff}}(Q)$")
ax[1].set_xlabel(r"$\mu$ or $Q$ (GeV)"); ax[1].set_ylabel(r"$1/\alpha$"); ax[1].legend(fontsize=8); ax[1].set_title("the electric charge, electron loop only", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] int_0^1 log(x(1 - x)) dx = -2 (the MOM conversion constant):", bool(integrate(log(var('x') * (1 - var('x'))), var('x'), 0, 1) == -2))
print("[PASS] at Q = 91.19 GeV the effective and MS-bar inverse couplings differ by 5/(9 pi) to 1e-3:", abs(inv_eff(91.19) - (1 / al - 2 / (3 * np.pi) * np.log(91.19 / me)) - 5 / (9 * np.pi)) < 1e-3)
print("[PASS] 2 int_0^1 x(1 - x) log(x(1 - x)) dx = -5/9:", bool(2 * integrate(var('x') * (1 - var('x')) * log(var('x') * (1 - var('x'))), var('x'), 0, 1) == -5 / 9))
print("wrote", os.path.relpath(OUT))
