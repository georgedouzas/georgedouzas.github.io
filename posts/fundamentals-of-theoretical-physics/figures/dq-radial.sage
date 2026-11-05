# Figure: the radial density G^2 + F^2 of the Dirac ground state 1s_{1/2} in a Coulomb potential -Z alpha/r, in units
# of the Bohr radius a_Z = 1/(m Z alpha), for Z = 1, 60, 92, against the Schrodinger density 4 rho^2 e^{-2 rho}. The
# relativistic state is pulled inwards and is singular like r^{2 gamma - 2} at the origin, gamma = sqrt(1 - Z^2 alpha^2).
# Output: figures/dq-radial.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-radial.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from math import gamma as Gam
al = 1 / 137.035999166
R = np.linspace(1e-4, 5, 500)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(R, 4 * R**2 * np.exp(-2 * R), color="black", lw=2.5, label="Schrodinger")
for Zn, col in [(1, "royalblue"), (60, "darkgreen"), (92, "firebrick")]:
    g = float(sqrt(1 - (Zn * al)**2))
    ax.plot(R, 2**(2 * g + 1) * R**(2 * g) * np.exp(-2 * R) / Gam(2 * g + 1), color=col, lw=1.4, ls="--", label=r"Dirac, $Z = %d$" % Zn)
ax.set_xlabel(r"$r/a_Z$"); ax.set_ylabel(r"radial density $G^2 + F^2$"); ax.set_ylim(0, 0.75)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT)
# checks: G = r^gamma e^{-lambda r}, F = ((1 - gamma)/(Z alpha)) G solve the radial equations with kappa = 1
r, m, a = var("r m a", domain="positive")   # a = Z alpha
gm = sqrt(1 - a^2); lam = m * a; E = m * gm; V = -a / r; kap = 1
G = r^gm * exp(-lam * r); F = (1 - gm) / a * G
eq1 = diff(G, r) - kap * G / r + (E - V + m) * F
eq2 = diff(F, r) + kap * F / r - (E - V - m) * G
print("[PASS] the ground state solves G' - kappa G/r = -(E - V + m) F, F' + kappa F/r = (E - V - m) G:", bool((eq1 / G).simplify_full() == 0) and bool((eq2 / G).simplify_full() == 0))
rho = var("rho", domain="positive"); g = var("g", domain="positive")
P = 2^(2 * g + 1) * rho^(2 * g) * exp(-2 * rho) / gamma(2 * g + 1)
nint = lambda f: numerical_integral(f, 0, oo)[0]
ok = all(abs(nint(P.subs(g=gv)) - 1) < 1e-9 and abs(nint((rho * P).subs(g=gv)) - (2 * gv + 1) / 2) < 1e-9 for gv in (0.3, 0.75, 1.0))
print("[PASS] the density is normalised and <r/a_Z> = (2 gamma + 1)/2 (gamma = 0.3, 0.75, 1; 3/2 at gamma = 1):", ok)
print("[PASS] E/m = gamma and <beta> = (1 - f^2)/(1 + f^2) = gamma for f = F/G = (1 - gamma)/(Z alpha):", bool((((1 - ((1 - gm) / a)^2) / (1 + ((1 - gm) / a)^2)) - gm).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
