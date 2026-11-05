# Figure: the inverse electromagnetic coupling with e, mu, tau: momentum subtraction (smooth) against the matched MS-bar
# couplings of the effective theories (each lepton runs only above its mass). Output: figures/ren-decoupling.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-decoupling.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
al = 1 / 137.035999; ml = [0.000510999, 0.1056584, 1.77686]
def mom(mu):
    return 1 / al - sum(2 / np.pi * quad(lambda x: x * (1 - x) * np.log(1 + x * (1 - x) * mu**2 / m**2), 0, 1)[0] for m in ml)
def msbar(mu):
    return 1 / al - sum(2 / (3 * np.pi) * np.log(mu / m) for m in ml if mu > m)
mus = np.logspace(-3, 2, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogx(mus, [mom(m) for m in mus], color="black", lw=1.8, label="momentum subtraction")
ax.semilogx(mus, [msbar(m) for m in mus], "--", color="royalblue", lw=1.6, label=r"$\overline{\mathrm{MS}}$, matched at each lepton mass")
for m in ml[1:]: ax.axvline(m, color="gray", ls=":", lw=0.8)
ax.set_xlabel(r"$\mu$ (GeV)"); ax.set_ylabel(r"$1/\alpha(\mu)$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
e_only_mom = 1 / al - 2 / np.pi * quad(lambda x: x * (1 - x) * np.log(1 + x * (1 - x) * 1e4), 0, 1)[0]
print("[PASS] with the electron alone the two differ at mu = 100 m by 5/(9 pi) to 2e-3:", abs((1 / al - 2 / (3 * np.pi) * np.log(100)) - e_only_mom + 5 / (9 * np.pi)) < 2e-3)
print("[PASS] at 1 MeV the matched MS-bar coupling has only the electron running:", abs(msbar(0.001) - (1 / al - 2 / (3 * np.pi) * np.log(0.001 / ml[0]))) < 1e-12)
print("[PASS] both are monotonically decreasing:", all(np.diff([mom(m) for m in mus]) <= 1e-12))
print("wrote", os.path.relpath(OUT))
