# Figure: the two branches E = +-sqrt(p^2 + m^2) of the Dirac Hamiltonian H = alpha.p + beta m (m = 1), separated by the
# gap 2m, with the non-relativistic parabola m + p^2/2m for comparison. Output: figures/dq-dispersion.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-dispersion.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
P = np.linspace(-3, 3, 400)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(P, np.sqrt(P**2 + 1), color="royalblue", lw=2, label=r"$E = +\sqrt{p^2 + m^2}$")
ax.plot(P, -np.sqrt(P**2 + 1), color="firebrick", lw=2, label=r"$E = -\sqrt{p^2 + m^2}$")
ax.plot(P, 1 + P**2 / 2, color="gray", ls="--", lw=1.2, label=r"$m + p^2/2m$")
ax.plot(P, np.abs(P), color="black", ls=":", lw=1, label=r"light cone $|p|$")
ax.plot(P, -np.abs(P), color="black", ls=":", lw=1)
ax.annotate("", xy=(0.05, 1), xytext=(0.05, -1), arrowprops=dict(arrowstyle="<->", color="darkgreen"))
ax.text(0.15, 0, r"gap $2m$", color="darkgreen", va="center")
ax.set_ylim(-3.5, 3.5); ax.set_xlabel(r"momentum $p$ (units of $m$)"); ax.set_ylabel(r"energy $E$ (units of $m$)")
ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); fig.savefig(OUT)
# checks: the Dirac Hamiltonian squares to p^2 + m^2 and has the two eigenvalues +-E, each twice
s1 = matrix(SR, [[0,1],[1,0]]); s2 = matrix(SR, [[0,-I],[I,0]]); s3 = matrix(SR, [[1,0],[0,-1]]); Z = matrix(SR, 2, 2); one = identity_matrix(SR, 2)
beta = block_matrix([[Z, one], [one, Z]], subdivide=False)
alpha = [block_matrix([[-s, Z], [Z, s]], subdivide=False) for s in (s1, s2, s3)]
p1, p2, p3, m = var("p1 p2 p3 m")
H = p1 * alpha[0] + p2 * alpha[1] + p3 * alpha[2] + m * beta
print("[PASS] H^2 = (p^2 + m^2) 1 for the Weyl-representation alpha, beta:", bool((H * H - (p1^2 + p2^2 + p3^2 + m^2) * identity_matrix(SR, 4)).simplify_full() == 0))
Hn = H.subs(p1=2, p2=3, p3=6, m=24)
print("[PASS] for p = (2, 3, 6), m = 24 the eigenvalues are +-25, each twice:", sorted(Hn.eigenvalues()) == [-25, -25, 25, 25])
print("[PASS] the non-relativistic parabola is the second-order Taylor polynomial of sqrt(p^2 + m^2):", bool(taylor(sqrt(p3^2 + m^2), p3, 0, 2) == m + p3^2 / (2 * m)))
print("wrote", os.path.relpath(OUT))
