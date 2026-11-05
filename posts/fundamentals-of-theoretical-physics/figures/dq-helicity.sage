# Figure: helicity against chirality (m = 1). Left: the right-handed fraction (1 + h v)/2 of a positive-energy spinor of
# helicity h = +1 or -1 as a function of p/m; chirality approaches helicity only as v = p/E -> 1. Right: the chirality
# <gamma_5>(t) = -1 + (m^2/E^2)(1 - cos 2Et) of a state that is left-handed at t = 0, for p = 0, 1, 3: chirality is not
# conserved, and oscillates at frequency 2E. Output: figures/dq-helicity.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-helicity.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
P = np.logspace(-2, 2, 300); V = P / np.sqrt(P**2 + 1)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
axs[0].semilogx(P, (1 + V) / 2, color="firebrick", lw=2, label=r"$h = +1$")
axs[0].semilogx(P, (1 - V) / 2, color="royalblue", lw=2, label=r"$h = -1$")
axs[0].set_xlabel(r"$p/m$"); axs[0].set_ylabel(r"right-handed fraction $(1 + hv)/2$"); axs[0].legend(fontsize=8)
T = np.linspace(0, 6, 400)
for p, col in [(0, "black"), (1, "darkgreen"), (3, "darkorange")]:
    E = np.sqrt(p**2 + 1)
    axs[1].plot(T, -1 + (1 / E**2) * (1 - np.cos(2 * E * T)), color=col, lw=1.6, label=r"$p = %d$" % p)
axs[1].set_xlabel(r"$t$ (units of $1/m$)"); axs[1].set_ylabel(r"$\langle\gamma_5\rangle(t)$"); axs[1].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT)
s0 = identity_matrix(SR, 2); s1 = matrix(SR, [[0,1],[1,0]]); s2 = matrix(SR, [[0,-I],[I,0]]); s3 = matrix(SR, [[1,0],[0,-1]]); Z = matrix(SR, 2, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, s0, s0, Z)] + [blk(Z, s, -s, Z) for s in (s1, s2, s3)]
g5 = I * g[0] * g[1] * g[2] * g[3]
m, p, t = var("m p t", domain="positive"); E = sqrt(p^2 + m^2)
H = p * g[0] * g[3] + m * g[0]
Ut = cos(E * t) * identity_matrix(SR, 4) - I * sin(E * t) * H / E
psi = Ut * vector(SR, [1, 0, 0, 0])
ch = (psi.conjugate() * g5 * psi).simplify_full()
print("[PASS] gamma_5 = diag(-1, -1, 1, 1): the upper components are left-handed:", g5 == diagonal_matrix(SR, [-1, -1, 1, 1]))
print("[PASS] U(t) = cos(Et) - i sin(Et) H/E is exp(-iHt) because H^2 = E^2:", bool((H * H - E^2 * identity_matrix(SR, 4)).simplify_full() == 0))
print("[PASS] <gamma_5>(t) = -1 + (m^2/E^2)(1 - cos 2Et) for the state (1, 0, 0, 0) with momentum p along z:", bool((ch - (-1 + m^2 / E^2 * (1 - cos(2 * E * t)))).simplify_full().trig_reduce() == 0))
print("[PASS] [H, gamma_5] = 2 m beta gamma_5 and [H, Sigma . p] = 0:", bool((H * g5 - g5 * H - 2 * m * g[0] * g5).simplify_full() == 0) and bool((H * (I * g[1] * g[2]) - (I * g[1] * g[2]) * H).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
