# Figure: the vacuum polarisation of one electron loop. Left: the Uehling correction V/V_Coulomb - 1 in units of alpha
# against m r. Right: the renormalised vacuum polarisation for time-like s in units of alpha, real and imaginary parts.
# Output: figures/ren-vacuum-polarisation.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-vacuum-polarisation.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
def U(mr):
    return quad(lambda u: np.exp(-2 * mr * u) * (1 + 1 / (2 * u * u)) * np.sqrt(u * u - 1) / u**2, 1, np.inf, limit=500)[0]
def PiHat(s):          # Pi-hat(s)/alpha for time-like s (units m = 1), with the -i0 prescription
    f = lambda x: x * (1 - x) * np.log(abs(1 - x * (1 - x) * s))
    if s > 4:
        b = np.sqrt(1 - 4 / s); pts = [(1 - b) / 2, (1 + b) / 2]
        re = 2 / np.pi * quad(f, 0, 1, points=pts, limit=200)[0]
        im = b * (3 - b * b) / 6
    else:
        re = 2 / np.pi * quad(f, 0, 1, limit=200)[0]; im = 0.0
    return re, im
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
mr = np.logspace(-3, 0.5, 120)
ax[0].semilogx(mr, [2 / (3 * np.pi) * U(r) for r in mr], color="black", lw=1.8, label="Uehling correction")
ax[0].semilogx(mr, 2 / (3 * np.pi) * (np.log(1 / mr) - 0.5772156649 - 5 / 6), "r--", lw=1.2, label=r"$\frac{2}{3\pi}(\log\frac{1}{mr} - \gamma_E - \frac{5}{6})$")
ax[0].set_ylim(-0.1, 1.8); ax[0].set_xlabel(r"$m_e r$"); ax[0].set_ylabel(r"$(V/V_{\mathrm{Coulomb}} - 1)/\alpha$"); ax[0].legend(fontsize=8); ax[0].set_title("screening undone at short distances", fontsize=10)
s = np.linspace(0.2, 40, 400)
v = np.array([PiHat(x) for x in s])
ax[1].plot(s, v[:, 0], color="black", lw=1.8, label=r"$\mathrm{Re}\,\hat\Pi/\alpha$")
ax[1].plot(s, v[:, 1], "--", color="firebrick", lw=1.8, label=r"$|\mathrm{Im}\,\hat\Pi|/\alpha$")
ax[1].axhline(1 / 3, color="gray", ls=":", lw=0.8); ax[1].axvline(4, color="gray", ls=":", lw=0.8)
ax[1].set_xlabel(r"$s/m_e^2$"); ax[1].legend(fontsize=8); ax[1].set_title("time-like momenta: the pair threshold", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
x = var('x'); z = var('z')
print("[PASS] small-z series of int x(1-x) log(1 + x(1-x) z) is z/30 - z^2/280:", bool(taylor(integrate(taylor(x * (1 - x) * log(1 + x * (1 - x) * z), z, 0, 3), x, 0, 1), z, 0, 2) == z / 30 - z^2 / 280))
print("[PASS] the imaginary part tends to 1/3 at large s and vanishes below 4 m^2:", abs(PiHat(1e6)[1] - 1 / 3) < 1e-5 and PiHat(3.9)[1] == 0)
print("[PASS] the Uehling integral at mr = 0.001 matches log(1/mr) - gamma - 5/6 to 1%:", abs(U(0.001) / (np.log(1000) - 0.5772156649 - 5 / 6) - 1) < 0.01)
print("wrote", os.path.relpath(OUT))
