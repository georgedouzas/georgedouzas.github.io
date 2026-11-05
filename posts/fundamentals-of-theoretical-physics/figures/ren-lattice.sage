# Figure: the lattice as a regulator (a = 1). Left: the lattice Laplacian 4 sin^2(k/2) against k^2. Right: the naive
# lattice fermion sin k, with a doubler at k = pi, and the Wilson mass r (1 - cos k), r = 1. Output: figures/ren-lattice.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-lattice.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
k = np.linspace(-np.pi, np.pi, 400)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(k, 4 * np.sin(k / 2)**2, color="royalblue", lw=2, label=r"lattice $\hat k^2 = 4\sin^2(k/2)$")
ax[0].plot(k, k**2, "k--", lw=1.2, label="continuum $k^2$")
ax[0].set_xlabel("$ka$ (Brillouin zone)"); ax[0].legend(fontsize=8); ax[0].set_title("scalar: bounded lattice symbol", fontsize=10)
ax[1].plot(k, np.abs(np.sin(k)), color="firebrick", lw=2, label=r"naive $|\sin ka|/a$")
ax[1].plot(k, 1 - np.cos(k), color="darkgreen", lw=2, label=r"Wilson mass $\frac{r}{a}(1 - \cos ka)$")
ax[1].plot(k, np.sqrt(np.sin(k)**2 + (1 - np.cos(k))**2), color="black", lw=1.2, ls="--", label="Wilson energy at $m = 0$")
ax[1].set_xlabel("$ka$"); ax[1].legend(fontsize=8); ax[1].set_title("fermion: the doubler at $ka = \\pi$", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
var('kk aa')
print("[PASS] 4/a^2 sin^2(ka/2) = k^2 - a^2 k^4/12 + a^4 k^6/360 + O(a^6):", bool(taylor(4 / aa**2 * sin(kk * aa / 2)**2, aa, 0, 4) == kk**2 - aa**2 * kk**4 / 12 + aa**4 * kk**6 / 360))
print("[PASS] the naive fermion symbol vanishes at k = 0 and k = pi, the Wilson mass there is 0 and 2r:", bool(sin(0) == 0 and sin(pi) == 0 and 1 - cos(pi) == 2))
import mpmath as mpm
C4 = mpm.quad(lambda u: (mpm.exp(-2 * u) * mpm.besseli(0, 2 * u))**4, [0, 1, 10, 100, mpm.inf])
print("[PASS] the four-dimensional lattice tadpole constant C_4 = 0.154933:", abs(C4 - 0.154933) < 1e-6)
print("wrote", os.path.relpath(OUT))
