# Figure: the zero-dimensional quartic theory: exact <phi^2>, first-order perturbation theory, the Hartree solution of
# the truncated Dyson-Schwinger equation and the two-loop truncation, against lambda. Output: figures/ren-dse.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-dse.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
import mpmath as mpm
def exact(l):
    f = lambda p: mpm.exp(-p**2 / 2 - l * p**4 / 24)
    return float(mpm.quad(lambda p: p**2 * f(p), [-mpm.inf, 0, mpm.inf]) / mpm.quad(f, [-mpm.inf, 0, mpm.inf]))
def two_loop(l):
    r = [x.real for x in np.roots([-l**2 / 6, 0, l / 2, 1, -1]) if abs(x.imag) < 1e-10 and x.real > 0]
    r = [x for x in r if x <= 1.0 + 1e-12]
    return max(r) if r else np.nan
ls = np.linspace(0.01, 10, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ls, [exact(l) for l in ls], color="black", lw=2, label="exact")
ax.plot(ls, 1 - ls / 2, ":", color="gray", lw=1.4, label=r"first order $1 - \lambda/2$")
ax.plot(ls, (np.sqrt(1 + 2 * ls) - 1) / ls, "--", color="royalblue", lw=1.6, label="Hartree")
ax.plot(ls, [two_loop(l) for l in ls], "-.", color="firebrick", lw=1.6, label="two-loop truncation")
ax.set_ylim(0, 1.05); ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\langle\phi^2\rangle$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
var('lam')
print("[PASS] the Hartree solution solves G^-1 = 1 + lam G/2:", bool(((1 / ((sqrt(1 + 2 * lam) - 1) / lam)) - 1 - lam / 2 * (sqrt(1 + 2 * lam) - 1) / lam).simplify_full() == 0))
print("[PASS] exact <phi^2> at lambda = 10 is 0.3915 and Hartree 0.3583:", abs(exact(10) - 0.3915) < 1e-4 and abs((sqrt(21.) - 1) / 10 - 0.3583) < 1e-4)
print("[PASS] the two-loop truncation has no real root at lambda = 10:", np.isnan(two_loop(10.0)))
print("wrote", os.path.relpath(OUT))
