# Figure: Compton scattering. Wavelength shift against the angle, and scattered photon energy for three incident
# energies. Output: figures/sr-compton.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-compton.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
lc = 2.42631; m = 0.511; th = np.linspace(0, np.pi, 300)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(np.degrees(th), lc*(1 - np.cos(th)), color="royalblue", lw=2); ax[0].set_xlabel(r"scattering angle $\theta$ (degrees)"); ax[0].set_ylabel(r"$\lambda' - \lambda$ (pm)")
for w, col in [(0.1, "seagreen"), (0.511, "royalblue"), (2.0, "darkorange")]:
    ax[1].plot(np.degrees(th), w/(1 + w/m*(1 - np.cos(th))), color=col, lw=2, label=r"$\omega = %s$ MeV" % w)
ax[1].set_xlabel(r"scattering angle $\theta$ (degrees)"); ax[1].set_ylabel(r"$\omega'$ (MeV)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
h, me, c = 6.62607015e-34, 9.1093837e-31, 2.99792458e8
print("[PASS] the Compton wavelength h/(m c) is 2.426 pm:", abs(h/(me*c)*1e12 - lc) < 1e-4)
w = 2.0; print("[PASS] wavelength form and energy form agree at 2 MeV, 90 degrees:", abs(1/(w/(1 + w/m)) - 1/w - 1/m) < 1e-12)
