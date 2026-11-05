# Figure: the mass shell E = sqrt(p^2 + m^2) for the electron, muon and proton, the massless line E = p and the
# Newtonian approximation for the muon. Output: figures/sr-mass-shell.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-mass-shell.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
p = np.linspace(0, 1500, 400)
fig, ax = plt.subplots(figsize=(6, 4.5))
for m, lab, col in [(0.511, "electron", "seagreen"), (105.66, "muon", "royalblue"), (938.27, "proton", "darkorange")]:
    ax.plot(p, np.sqrt(p**2 + m**2), color=col, lw=2, label=lab)
ax.plot(p, p, "k--", lw=1, label="massless, $E = p$")
pm = p[p < 400]; ax.plot(pm, 105.66 + pm**2/(2*105.66), "b:", lw=1.5, label="muon, Newtonian $m + p^2/2m$")
ax.set_xlabel("momentum $p$ (MeV)"); ax.set_ylabel("energy $E$ (MeV)"); ax.set_ylim(0, 1800); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] every curve lies above the massless line:", all(np.all(np.sqrt(p**2 + m**2) >= p) for m in (0.511, 105.66, 938.27)))
