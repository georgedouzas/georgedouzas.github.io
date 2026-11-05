# Figure: photon-photon scattering below the electron mass: the Euler-Heisenberg cross section
# (15568/10125) alpha^4 omega^6/(16 pi m^8) against the dimension-four estimate alpha^4/(16 pi omega^2), units m_e = 1.
# Output: figures/ren-euler-heisenberg.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-euler-heisenberg.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
al = 1 / 137.035999
w = np.logspace(-6, -0.5, 200)
eh = 15568 / 10125 * al**4 * w**6 / (16 * np.pi)
naive = al**4 / (16 * np.pi * w**2)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(w, eh, color="black", lw=1.8, label="Euler-Heisenberg ($D = 8$)")
ax.loglog(w, naive, "--", color="firebrick", lw=1.6, label=r"dimension four, $\alpha^4/16\pi\omega^2$")
ax.axvline(1 / 0.51099895e6, color="gray", ls=":", lw=1); ax.text(1.3 / 0.511e6, 1e-30, r"$\omega = 1\,$eV", fontsize=8)
ax.set_xlabel(r"$\omega/m_e$"); ax.set_ylabel(r"$\sigma\,m_e^2$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
x = 1 / 0.51099895e6
print("[PASS] the ratio of the two estimates is 1.5375 (omega/m)^8:", abs((15568 / 10125 * x**6) / (1 / x**2) / (1.5375 * x**8) - 1) < 1e-3)
print("[PASS] at omega = 1 eV the ratio is about 3e-46:", 2e-46 < 15568 / 10125 * x**8 < 4e-46)
print("[PASS] 15568/10125 = 1.5376:", abs(15568 / 10125 - 1.5376) < 1e-4)
print("wrote", os.path.relpath(OUT))
