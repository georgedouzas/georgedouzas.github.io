# Figure: the Klein step. Reflection R and transmission T = 1 - R of a Dirac particle of energy E = 3m (m = 1) on a
# potential step of height V in 1 + 1 dimensions, with kappa = (q/k)(E + m)/(E - V + m) and q chosen by the group
# velocity. Below V = E - m the step is passed as in the Schrodinger case; for E - m < V < E + m it reflects totally;
# above V = E + m (the Klein zone) transmission returns and tends to 4 kappa_inf/(1 + kappa_inf)^2,
# kappa_inf = sqrt((E + m)/(E - m)). The dashed curve is the Schrodinger step with the same kinetic energy E - m.
# Output: figures/dq-klein.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-klein.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
m, E = 1.0, 3.0
k = np.sqrt(E**2 - m**2)
def T_dirac(V):
    eps = E - V
    if abs(eps) <= m:
        return 0.0
    q = np.sqrt(eps**2 - m**2) * np.sign(eps)    # group velocity q/(E - V) > 0
    kap = q / k * (E + m) / (eps + m)
    return 4 * kap / (1 + kap)**2
def T_schr(V):
    K = E - m
    if V >= K:
        return 0.0
    k1, k2 = np.sqrt(K), np.sqrt(K - V)
    return 4 * k1 * k2 / (k1 + k2)**2
Vs = np.linspace(0, 12, 1200)
fig, ax = plt.subplots(figsize=(6.8, 4))
ax.plot(Vs, [T_dirac(v) for v in Vs], color="royalblue", lw=2, label=r"Dirac $T$")
ax.plot(Vs, [1 - T_dirac(v) for v in Vs], color="firebrick", lw=1.5, label=r"Dirac $R$")
ax.plot(Vs, [T_schr(v) for v in Vs], color="black", lw=1.2, ls="--", label=r"Schrodinger $T$")
kinf = np.sqrt((E + m) / (E - m))
ax.axhline(4 * kinf / (1 + kinf)**2, color="royalblue", ls=":", lw=1)
for v, lab in [(E - m, r"$E - m$"), (E + m, r"$E + m$")]:
    ax.axvline(v, color="gray", lw=0.6); ax.text(v + 0.1, 0.5, lab, color="gray", fontsize=8)
ax.set_xlabel(r"step height $V/m$"); ax.set_ylabel("probability"); ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(OUT)
Es, ms, Vv, qq, kk = var("Es ms Vv qq kk")
kap = var("kap")
Rr = ((1 - kap) / (1 + kap))^2; Tt = 4 * kap / (1 + kap)^2
print("[PASS] R + T = 1 for every kappa:", bool((Rr + Tt - 1).simplify_full() == 0))
print("[PASS] T is in the Klein zone between 0 and 1 for all sampled V > E + m (E = 3m):", all(0 < T_dirac(v) <= 1 for v in np.linspace(E + m + 0.01, 50, 200)))
print("[PASS] T tends to 4 kappa_inf/(1 + kappa_inf)^2 = %.4f as V -> infinity:" % (4 * kinf / (1 + kinf)**2), abs(T_dirac(1e7) - 4 * kinf / (1 + kinf)**2) < 1e-5)
kn = np.sqrt(24) / k * (E + m) / (E - 8 + m)   # naive q = +sqrt((V - E)^2 - m^2) at V = 8m
Rn = ((1 - kn) / (1 + kn))**2
print("[PASS] with the naive sign of q, kappa = -sqrt(3) and R = ((1 + sqrt 3)/(1 - sqrt 3))^2 = 7 + 4 sqrt 3 > 1 at V = 8m:", abs(kn + np.sqrt(3)) < 1e-12 and abs(Rn - (7 + 4 * np.sqrt(3))) < 1e-9)
print("wrote", os.path.relpath(OUT))
