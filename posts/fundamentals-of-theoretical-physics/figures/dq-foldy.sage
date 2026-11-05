# Figure: the free Foldy-Wouthuysen transformation (m = 1). Left: the exact energy sqrt(p^2 + m^2) and its expansions
# m + p^2/2m and m + p^2/2m - p^4/8m^3, the first terms of the Foldy-Wouthuysen series. Right: the angle theta(p) of the
# exact transformation U = exp(beta alpha.p theta/|p|), tan 2 theta = |p|/m, which tends to pi/4.
# Output: figures/dq-foldy.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-foldy.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
P = np.linspace(0, 2, 300)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
axs[0].plot(P, np.sqrt(P**2 + 1), color="black", lw=2, label=r"$\sqrt{p^2 + m^2}$")
axs[0].plot(P, 1 + P**2 / 2, color="royalblue", lw=1.5, ls="--", label=r"$m + p^2/2m$")
axs[0].plot(P, 1 + P**2 / 2 - P**4 / 8, color="firebrick", lw=1.5, ls="-.", label=r"$\dots - p^4/8m^3$")
axs[0].set_ylim(0.9, 2.6); axs[0].set_xlabel(r"$p/m$"); axs[0].set_ylabel(r"$E/m$"); axs[0].legend(fontsize=8)
P2 = np.linspace(0, 10, 300)
axs[1].plot(P2, np.arctan(P2) / 2, color="darkgreen", lw=2)
axs[1].axhline(np.pi / 4, color="gray", ls=":", lw=1); axs[1].text(6, np.pi / 4 - 0.07, r"$\pi/4$", color="gray")
axs[1].set_xlabel(r"$p/m$"); axs[1].set_ylabel(r"$\theta(p) = \frac{1}{2}\arctan(p/m)$")
fig.tight_layout(); fig.savefig(OUT)
# checks in the Dirac representation, momentum along z
one = identity_matrix(SR, 2); Z = matrix(SR, 2, 2); s3 = matrix(SR, [[1,0],[0,-1]])
beta = block_matrix([[one, Z], [Z, -one]], subdivide=False); a3 = block_matrix([[Z, s3], [s3, Z]], subdivide=False)
p, m = var("p m", domain="positive"); E = sqrt(p^2 + m^2)
th = arctan(p / m) / 2
Uf = cos(th) * identity_matrix(SR, 4) + sin(th) * beta * a3
H = p * a3 + m * beta
HFW = Uf * H * Uf.conjugate_transpose()
print("[PASS] (beta alpha_3)^2 = -1, so exp(theta beta alpha_3) = cos theta + beta alpha_3 sin theta:", (beta * a3)^2 == -identity_matrix(SR, 4))
print("[PASS] U H U^dag = beta sqrt(p^2 + m^2) for tan 2 theta = p/m (at three momenta):", all((HFW - E * beta).subs(p=a, m=b).n().norm() < 1e-12 for a, b in [(0.3, 1), (2, 0.5), (7, 3)]))
print("[PASS] the expansion of sqrt(p^2 + m^2) is m + p^2/2m - p^4/8m^3 + p^6/16m^5:", bool(taylor(E, p, 0, 6) == m + p^2 / (2 * m) - p^4 / (8 * m^3) + p^6 / (16 * m^5)))
print("wrote", os.path.relpath(OUT))
