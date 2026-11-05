# Figure: a charge in a uniform magnetic field. Angle swept against coordinate time for three Lorentz factors (m = q = B
# = 1), and the cyclotron frequency of protons in 1 T against the kinetic energy. Output: figures/sr-cyclotron.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-cyclotron.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
t = np.linspace(0, 30, 400)
for g, col in [(1.0, "seagreen"), (2.0, "royalblue"), (4.0, "darkorange")]:
    ax[0].plot(t, t/g, color=col, lw=2, label=r"$\gamma = %s$" % g)
for k in range(1, 5): ax[0].axhline(2*np.pi*k, color="0.8", lw=0.6)
ax[0].set_xlabel("coordinate time $t$"); ax[0].set_ylabel("angle swept (rad)"); ax[0].legend(fontsize=8)
mp = 938.272; T = np.logspace(-1, 4, 300); qB_over_m = 1.602176634e-19*1.0/1.67262192e-27
ax[1].semilogx(T, qB_over_m/(1 + T/mp)/(2*np.pi)/1e6, color="royalblue", lw=2)
ax[1].set_xlabel("proton kinetic energy (MeV)"); ax[1].set_ylabel(r"cyclotron frequency $\omega/2\pi$ in 1 T (MHz)")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the non-relativistic proton cyclotron frequency in 1 T is 15.25 MHz:", abs(qB_over_m/(2*np.pi)/1e6 - 15.245) < 0.01)
