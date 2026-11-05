# Figure: the Casimir energy of a real scalar on R^{1,3} x S^1 (circumference l) in units of 3 zeta(5)/(4 pi^2 l^4) against m l,
# periodic and antiperiodic, from the polylogarithm formula (Ponton-Poppitz eq. 18). Output: figures/ren-casimir.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-casimir.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
import mpmath as mpm
z5 = float(zeta(5))
def comb(x, s):
    q = s * mpm.e**(-x)
    return float(mpm.polylog(5, q) + x * mpm.polylog(4, q) + x**2 / 3 * mpm.polylog(3, q))
xs = np.linspace(0.001, 8, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(xs, [-comb(x, 1) / z5 for x in xs], color="royalblue", lw=1.8, label="periodic")
ax.plot(xs, [-comb(x, -1) / z5 for x in xs], color="firebrick", lw=1.8, label="antiperiodic")
ax.axhline(0, color="gray", lw=0.7)
ax.set_xlabel(r"$m\ell$"); ax.set_ylabel(r"$V\,/\,\frac{3\zeta(5)}{4\pi^2\ell^4}$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the massless periodic value is -1 and the antiperiodic +15/16 in these units:", abs(-comb(1e-9, 1) / z5 + 1) < 1e-6 and abs(-comb(1e-9, -1) / z5 - 15 / 16) < 1e-6)
print("[PASS] zeta'(-4) = 3 zeta(5)/(4 pi^4):", abs(float(diff(zeta(var('s')), var('s')).subs(s=-4)) - 3 * z5 / (4 * float(pi)**4)) < 1e-14)
print("[PASS] the polylogarithm combination equals the winding sum at m l = 1:", abs(comb(1.0, 1) - sum(np.exp(-k) * (3 + 3 * k + k * k) / (3 * k**5) for k in range(1, 60))) < 1e-12)
print("wrote", os.path.relpath(OUT))
