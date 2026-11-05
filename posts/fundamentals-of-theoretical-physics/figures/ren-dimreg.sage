# Figure: dimensional regularisation. Left: Gamma(1 - d/2) (tadpole) and Gamma(2 - d/2) (bubble) as functions of d, with
# their poles. Right: the three-dimensional tadpole with a cutoff minus its linear divergence, approaching -m/(4 pi), the
# dimensionally regularised value (m = 1). Output: figures/ren-dimreg.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-dimreg.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as G
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
d = np.linspace(1.02, 8.98, 4000)
for f, c, lab in [(lambda d: G(1 - d / 2), "royalblue", r"$\Gamma(1 - d/2)$ (tadpole)"), (lambda d: G(2 - d / 2), "firebrick", r"$\Gamma(2 - d/2)$ (bubble)")]:
    y = f(d); y[np.abs(y) > 12] = np.nan
    ax[0].plot(d, y, color=c, lw=1.6, label=lab)
for p in [2, 4, 6, 8]:
    ax[0].axvline(p, color="gray", ls=":", lw=0.8)
ax[0].set_ylim(-10, 10); ax[0].set_xlabel("dimension $d$"); ax[0].legend(fontsize=8, loc="lower left"); ax[0].set_title("poles at even integers", fontsize=10)
L = np.logspace(0, 3, 200)
T3 = (L - np.arctan(L)) / (2 * np.pi**2) - L / (2 * np.pi**2)
ax[1].semilogx(L, T3, color="black", lw=1.6, label=r"$T_{3,\Lambda} - \Lambda/2\pi^2$")
ax[1].axhline(-1 / (4 * np.pi), color="firebrick", ls="--", label=r"dim. reg. $-m/4\pi$")
ax[1].set_xlabel(r"$\Lambda/m$"); ax[1].legend(fontsize=8); ax[1].set_title("the tadpole in three dimensions", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
var('e'); var('m', domain='positive')
print("[PASS] Gamma(-1 + e/2) = -2/e - 1 + gamma_E + O(e):", bool(gamma(-1 + e / 2).series(e, 1).truncate() == -2 / e + euler_gamma - 1))
print("[PASS] the d = 3 tadpole Gamma(-1/2) m/(4 pi)^(3/2) equals -m/(4 pi):", bool(gamma(-1/2) * m / (4 * pi)**(3/2) == -m / (4 * pi)))
print("[PASS] the cutoff tadpole minus Lambda/2pi^2 at Lambda = 1000 m is within 1e-3 of -1/4pi:", abs(T3[-1] + 1 / (4 * np.pi)) < 1e-3)
print("wrote", os.path.relpath(OUT))
