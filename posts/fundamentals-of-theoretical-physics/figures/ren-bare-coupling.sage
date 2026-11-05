# Figure: bare and renormalised couplings, b = 3/(16 pi^2). Left: lambda_0 mu^-eps = lambda + b lambda^2/eps against eps.
# Right: solutions of mu dlambda/dmu = -eps lambda + b lambda^2 from lambda = 2 for eps = 0, 0.03, 0.06. Output: figures/ren-bare-coupling.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-bare-coupling.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
b = 3 / (16 * np.pi**2)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
eps = np.linspace(0.005, 0.5, 300)
for lam, c in [(1, "royalblue"), (2, "darkorange"), (4, "firebrick")]:
    ax[0].plot(eps, lam + b * lam**2 / eps, color=c, lw=1.8, label=r"$\lambda = %d$" % lam)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\epsilon = 4 - d$"); ax[0].set_ylabel(r"$\lambda_0\mu^{-\epsilon}$"); ax[0].legend(fontsize=8); ax[0].set_title("bare coupling at fixed renormalised coupling", fontsize=10)
lam0 = 2.0
t = np.linspace(-6, 6, 300)                   # log(mu/mu0)
for e, c in [(0.0, "black"), (0.03, "royalblue"), (0.06, "firebrick")]:
    if e == 0:
        lam = lam0 / (1 - b * lam0 * t)
    else:
        ls = e / b                             # fixed point of -e lam + b lam^2
        lam = ls / (1 - (1 - ls / lam0) * np.exp(e * t))
    lam[(lam < 0) | (lam > 12)] = np.nan
    ax[1].plot(t, lam, color=c, lw=1.8, label=(r"$\epsilon = %g$" % e) + ("" if e == 0 else r", $\lambda^* = %.2f$" % (e / b)))
    if e > 0: ax[1].axhline(e / b, color=c, ls=":", lw=0.8)
ax[1].set_ylim(0, 8); ax[1].set_xlabel(r"$\log(\mu/\mu_0)$"); ax[1].set_ylabel(r"$\lambda(\mu)$"); ax[1].legend(fontsize=8); ax[1].set_title(r"running with $\beta = -\epsilon\lambda + b\lambda^2$", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
var('lam eps t0'); bb = 3 / (16 * pi^2)
lstar = eps / bb
sol = lstar / (1 - (1 - lstar / 2) * exp(eps * t0))
print("[PASS] lambda* / (1 - (1 - lambda*/lambda0) e^(eps t)) solves dlambda/dt = -eps lambda + b lambda^2:", bool((diff(sol, t0) - (-eps * sol + bb * sol^2)).simplify_full() == 0))
print("[PASS] it tends to the d = 4 one-loop running as eps -> 0:", bool(limit(sol, eps=0) == (2 / (1 - 2 * bb * t0)).simplify_full()) or bool((limit(sol, eps=0) - 2 / (1 - 2 * bb * t0)).simplify_full() == 0))
print("[PASS] lambda/(1 - b lambda/eps) has the poles b^k lambda^(k+1)/eps^k and beta = -eps lambda + b lambda^2 exactly:", bool((taylor(lam / (1 - bb * lam / eps), lam, 0, 3) - (lam + bb * lam^2 / eps + bb^2 * lam^3 / eps^2)).simplify_full() == 0) and bool(((-eps * lam / (1 - bb * lam / eps)) / diff(lam / (1 - bb * lam / eps), lam) - (-eps * lam + bb * lam^2)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
