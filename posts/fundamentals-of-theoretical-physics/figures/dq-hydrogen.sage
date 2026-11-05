# Figure: the Dirac spectrum of hydrogen. Left: the fine structure of n = 2 and n = 3, as offsets (GHz) of the Dirac
# levels (reduced mass included) from the Bohr levels; levels with equal n and j coincide. Right: the ground-state
# energy E_1/m = sqrt(1 - (Z alpha)^2) against Z, with the Schrodinger value 1 - (Z alpha)^2/2; the Dirac value ends at
# Z alpha = 1. Output: figures/dq-hydrogen.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-hydrogen.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
RF = RealField(120)
alpha = RF(1) / RF("137.035999166"); me = RF("510998.95069"); h = RF("4.135667696e-15"); Mp = RF("1836.15267343")
mr = me / (1 + 1 / Mp)
def f(n, j, Z=1):
    k = RF(j) + RF(1) / 2
    return (1 + (Z * alpha)^2 / (n - k + sqrt(k^2 - (Z * alpha)^2))^2)^(-RF(1) / 2)
E = lambda n, j: me + mr * (f(n, j) - 1)
Bohr = lambda n: me - mr * alpha^2 / (2 * n^2)
GHz = lambda dE: float(dE / h / 1e9)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
labels = {2: [("2S_{1/2}, 2P_{1/2}", 1/2), ("2P_{3/2}", 3/2)], 3: [("3S_{1/2}, 3P_{1/2}", 1/2), ("3P_{3/2}, 3D_{3/2}", 3/2), ("3D_{5/2}", 5/2)]}
for x0, n in [(0, 2), (1.6, 3)]:
    for lab, j in labels[n]:
        y = GHz(E(n, j) - Bohr(n))
        axs[0].hlines(y, x0, x0 + 1, color="royalblue", lw=2)
        axs[0].text(x0 + 1.05, y, r"$%s$" % lab, va="center", fontsize=7)
    axs[0].hlines(0, x0, x0 + 1, color="gray", ls=":", lw=1)
axs[0].text(0.2, 1.5, r"$n = 2$"); axs[0].text(1.8, 1.5, r"$n = 3$")
axs[0].set_xlim(-0.1, 3.6); axs[0].set_xticks([]); axs[0].set_ylabel("offset from the Bohr level (GHz)")
Zs = np.linspace(0, 137, 300); al = float(alpha)
axs[1].plot(Zs, np.sqrt(np.maximum(1 - (Zs * al)**2, 0)), color="firebrick", lw=2, label="Dirac")
axs[1].plot(Zs, 1 - (Zs * al)**2 / 2, color="black", lw=1.5, ls="--", label="Schrodinger")
axs[1].set_xlabel(r"$Z$"); axs[1].set_ylabel(r"$E_1/m$"); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] 2P_{3/2} - 2P_{1/2} from the Dirac formula with reduced mass is 10.94 GHz:", abs(GHz(E(2, 3/2) - E(2, 1/2)) - 10.9437) < 1e-3)
print("[PASS] 3D_{5/2} - 3D_{3/2} = 1.081 GHz and 3P_{3/2} - 3P_{1/2} = 3.243 GHz:", abs(GHz(E(3, 5/2) - E(3, 3/2)) - 1.0808) < 1e-3 and abs(GHz(E(3, 3/2) - E(3, 1/2)) - 3.2426) < 1e-3)
dirac12 = (E(2, 1/2) - E(1, 1/2)) / h
print("[PASS] the Dirac 1S-2S frequency exceeds the measured 2 466 061 413 187 035 Hz by 7.11 GHz:", abs(float(dirac12 - RF("2466061413187035")) / 1e9 - 7.106) < 1e-2)
a = var("a"); n, jj = var("n jj", domain="positive")
k = jj + 1/2
expan = taylor((1 + a^2 / (n - k + sqrt(k^2 - a^2))^2)^(-1/2), a, 0, 4)
print("[PASS] f(n, j) = 1 - a^2/(2n^2) - (a^4/(2n^4))(n/(j + 1/2) - 3/4) + O(a^6):", bool((expan - (1 - a^2 / (2 * n^2) - a^4 / (2 * n^4) * (n / k - 3/4))).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
