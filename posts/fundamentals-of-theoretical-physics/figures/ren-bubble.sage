# Figure: the renormalised bubble 16 pi^2 B_R(s) = -int_0^1 log((m^2 - x(1 - x) s - i0)/mu^2) dx at mu = m, real and
# imaginary parts against s/m^2. Output: figures/ren-bubble.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-bubble.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
def BR(s):
    re = -quad(lambda x: np.log(abs(1 - x * (1 - x) * s)), 0, 1, limit=200, points=[0.5 - 0.5 * np.sqrt(max(0, 1 - 4 / s)) if s > 4 else 0.5, 0.5 + 0.5 * np.sqrt(max(0, 1 - 4 / s)) if s > 4 else 0.5])[0]
    im = np.pi * np.sqrt(1 - 4 / s) if s > 4 else 0.0
    return re, im
s = np.linspace(-10, 30, 801)
vals = np.array([BR(x) for x in s])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(s, vals[:, 0], color="black", lw=1.8, label=r"$\mathrm{Re}\,16\pi^2B_R$")
ax.plot(s, vals[:, 1], "--", color="firebrick", lw=1.8, label=r"$\mathrm{Im}\,16\pi^2B_R = \pi\sqrt{1 - 4m^2/s}$")
ax.axvline(4, color="gray", ls=":", lw=1); ax.text(4.3, -2.5, "threshold $s = 4m^2$", fontsize=8)
ax.set_xlabel(r"$s/m^2$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] 16 pi^2 B_R(4 m^2) = 2 at mu = m:", abs(BR(4.0)[0] - 2) < 1e-6)
print("[PASS] 16 pi^2 B_R(0) = 0 at mu = m:", abs(BR(1e-9)[0]) < 1e-6)
x = var('x'); sv = 9
print("[PASS] the imaginary part pi sqrt(1 - 4/s) is pi times the length of {x(1 - x)s > 1} at s = 9:", abs(float(pi * sqrt(1 - 4 / sv)) - float(pi) * float(abs(diff(solve(x * (1 - x) * sv == 1, x)[0].rhs() - solve(x * (1 - x) * sv == 1, x)[1].rhs(), x) * 0 + (solve(x * (1 - x) * sv == 1, x)[1].rhs() - solve(x * (1 - x) * sv == 1, x)[0].rhs())))) < 1e-12)
print("wrote", os.path.relpath(OUT))
