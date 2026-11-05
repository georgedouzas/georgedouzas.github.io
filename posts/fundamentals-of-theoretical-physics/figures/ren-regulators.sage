# Figure: regulators compared on the Euclidean bubble B(p^2) of the quartic theory (m = 1). Left: the radial weight of
# B(0) with a sharp cutoff, a Pauli-Villars field and a proper-time factor. Right: the subtracted bubble 16 pi^2 (B(p^2) -
# B(0)) from the Feynman-parameter formula (Pauli-Villars and dimensional regularisation) and from a sharp cutoff at
# Lambda = 30 m (angular integral done in closed form, radial integral numerically). Output: figures/ren-regulators.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-regulators.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
m = 1.0
def sub_exact(p):
    return -quad(lambda x: np.log(1 + x * (1 - x) * p**2 / m**2), 0, 1)[0]
def B_cut(p, L):
    if p == 0:
        return 2 * quad(lambda l: l**3 / (l**2 + m**2)**2, 0, L, limit=200)[0]
    def f(l):
        A = l**2 + p**2 + m**2; Bq = 2 * l * p
        return l * (A - np.sqrt(A**2 - Bq**2)) / ((l**2 + m**2) * p**2)
    return quad(f, 0, L, limit=400, points=[p])[0]
# 16 pi^2 B = 2 * int l^3 ... /(...) for the radial measure d^4l/(2pi)^4 = l^3 dl/(8 pi^2)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
l = np.linspace(0, 40, 800)
w = l**3 / (l**2 + 1)**2
ax[0].plot(l, np.where(l < 10, w, 0), color="black", lw=1.5, label=r"sharp cutoff $\Lambda = 10m$")
M = 10.0
ax[0].plot(l, l**3 * (1 / (l**2 + 1)**2 - 1 / (l**2 + M**2)**2), color="royalblue", lw=1.5, label=r"Pauli-Villars $M = 10m$")
ax[0].plot(l, w * np.exp(-l**2 / 100), color="darkorange", lw=1.5, label=r"proper time $e^{-\ell^2/\Lambda^2}$, $\Lambda = 10m$")
ax[0].set_xlabel(r"$\ell/m$"); ax[0].set_ylabel(r"$\ell^3 \times$ integrand"); ax[0].legend(fontsize=8); ax[0].set_title("weight of the loop momentum in $B(0)$", fontsize=10)
ps = np.linspace(0.01, 8, 200)
ax[1].plot(ps, [sub_exact(p) for p in ps], color="royalblue", lw=2, label="Pauli-Villars / dim. reg.")
pd = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
L = 30.0
cut = [B_cut(p, L) - B_cut(0, L) for p in pd]
ax[1].plot(pd, cut, "ko", ms=4, label=r"sharp cutoff $\Lambda = 30m$")
ax[1].plot(ps[ps > 1.5], -np.log(ps[ps > 1.5]**2) + 2, "k--", lw=1, label=r"$-\log(p^2/m^2) + 2$")
ax[1].set_xlabel(r"$p/m$"); ax[1].set_ylabel(r"$16\pi^2(B(p^2) - B(0))$"); ax[1].legend(fontsize=8); ax[1].set_title("the subtracted bubble", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
var('x a')
print("[PASS] Pauli-Villars bubble at p = 0 equals log(M^2/m^2) (M = 10):", abs(2 * quad(lambda t: t**3 * (1 / (t**2 + 1)**2 - 1 / (t**2 + 100)**2), 0, np.inf)[0] - np.log(100)) < 1e-8)
print("[PASS] cutoff and exact subtracted bubbles agree to 2e-3 at the plotted points:", max(abs(c - sub_exact(p)) for c, p in zip(cut, pd)) < 2e-3)
print("[PASS] small-momentum series -a/6 + a^2/60 of the subtracted bubble:", bool(taylor(-integrate(taylor(log(1 + x * (1 - x) * a), a, 0, 3), x, 0, 1), a, 0, 2) == -a / 6 + a**2 / 60))
print("wrote", os.path.relpath(OUT))
