# Figure: renormalisation-group improvement of the symmetric-point amplitude of the massless quartic theory,
# lambda(mu0) = 4: fixed order at mu = mu0, one-loop improved (mu = Q) and two-loop improved, divided by lambda(mu0).
# Output: figures/ren-callan-symanzik.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-callan-symanzik.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
b = 3 / (16 * np.pi**2); l0 = 4.0
t = np.linspace(0, 10, 300)
fixed = l0 + b * l0**2 * t - b * l0**2
lam1 = l0 / (1 - b * l0 * t)
imp1 = lam1 - b * lam1**2
sol = solve_ivp(lambda tt, l: [b * l[0]**2 - 17 / 3 * l[0]**3 / (16 * np.pi**2)**2], [0, 10], [l0], t_eval=t, rtol=1e-10)
lam2 = sol.y[0]; imp2 = lam2 - b * lam2**2
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(t, fixed / l0, "k--", lw=1.4, label="fixed order, $\\mu = \\mu_0$")
ax.plot(t, imp1 / l0, color="royalblue", lw=1.8, label="improved, one-loop running")
ax.plot(t, imp2 / l0, ":", color="firebrick", lw=1.8, label="improved, two-loop running")
ax.set_xlabel(r"$\log(Q/\mu_0)$"); ax.set_ylabel(r"$-\mathcal{M}(Q)/\lambda(\mu_0)$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
var('lam tt bb')
G = lam - bb * lam^2 + bb * lam^2 * tt
print("[PASS] the fixed-order amplitude obeys (d/dt - beta d/dlambda) G = O(lambda^3) with beta = b lambda^2:", bool(taylor(diff(G, tt) - bb * lam^2 * diff(G, lam), lam, 0, 2) == 0))
print("[PASS] the improved amplitude re-expanded to order lambda^2 equals the fixed-order one:", bool(taylor((lam / (1 - bb * lam * tt)) - bb * (lam / (1 - bb * lam * tt))^2 - G, lam, 0, 2) == 0))
print("[PASS] at t = 10 the one-loop improved value exceeds fixed order:", imp1[-1] > fixed[-1])
print("wrote", os.path.relpath(OUT))
