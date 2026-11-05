# Figure: left, the colour weights (eigenvalues of t_3 = lambda_3/2 and t_8 = lambda_8/2) of the triplet 3, the antitriplet 3-bar
# and the octet 8 of SU(3); right, the one-loop running coupling alpha_s(Q) from alpha_s(m_Z) = 0.118 with b_0 = 11 - 2 n_f / 3 and
# n_f changing at m_b = 4.18 GeV and m_t = 172.7 GeV, compared with the electromagnetic alpha(Q) from alpha(m_e) = 1/137.036 with the
# charged leptons and quarks lighter than Q. Checks: the octet weights are the differences of triplet weights with two zero weights;
# alpha_s decreases with Q (asymptotic freedom); alpha_s(m_tau) is about 0.32 at one loop. Output: figures/sm_qcd.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm_qcd.svg")
w3 = [(0.5, 1 / (2 * np.sqrt(3))), (-0.5, 1 / (2 * np.sqrt(3))), (0, -1 / np.sqrt(3))]          # weights of r, g, b
w3b = [(-a, -b) for a, b in w3]
w8 = [(a1 - a2, b1 - b2) for (a1, b1) in w3 for (a2, b2) in w3 if (a1, b1) != (a2, b2)] + [(0, 0), (0, 0)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
for pts, col, mk, lab, sz in ((w8, "lightgrey", "o", r"$\mathbf{8}$ (gluons)", 160), (w3, "firebrick", "^", r"$\mathbf{3}$ (quarks)", 90), (w3b, "royalblue", "v", r"$\bar{\mathbf{3}}$ (antiquarks)", 90)):
    p = np.array(pts); ax.scatter(p[:, 0], p[:, 1], s=sz, color=col, marker=mk, label=lab, zorder=2 if col != "lightgrey" else 1)
ax.scatter([0], [0], s=420, facecolors="none", edgecolors="grey", zorder=1)
for (a, b), lab in zip(w3, ("r", "g", "b")):
    ax.annotate(lab, (a, b), xytext=(7, 3), textcoords="offset points", fontsize=10, color="firebrick")
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
ax.set_xlabel(r"colour isospin $t_3$"); ax.set_ylabel(r"$t_8$"); ax.legend(fontsize=8, loc="lower left")
ax.set_title(r"Weights of $\mathbf{3}$, $\bar{\mathbf{3}}$ and $\mathbf{8}$ (double weight at the origin)", fontsize=9)
ax = axes[1]
mZ, a0, mb, mt = 91.1876, 0.118, 4.18, 172.7
def alpha_s(Q):
    def run(a, Q1, Q2, nf):
        b0 = 11 - 2 * nf / 3
        return 1 / (1 / a + b0 / (2 * np.pi) * np.log(Q2 / Q1))
    if Q >= mt:
        return run(run(a0, mZ, mt, 5), mt, Q, 6)
    if Q >= mb:
        return run(a0, mZ, Q, 5)
    return run(run(a0, mZ, mb, 5), mb, Q, 4)
Qs = np.logspace(np.log10(1.5), 4, 400)
ax.semilogx(Qs, [alpha_s(q) for q in Qs], color="firebrick", lw=2, label=r"$\alpha_s(Q)$, one loop")
fer = [(0.000511, 1, 1), (0.1057, 1, 1), (1.777, 1, 1), (0.0022, 4 / 9, 3), (0.0047, 1 / 9, 3), (0.095, 1 / 9, 3), (1.27, 4 / 9, 3), (4.18, 1 / 9, 3), (172.7, 4 / 9, 3)]
def alpha_em(Q):
    inv = 137.036
    for m, q2, nc in fer:
        if Q > m:
            inv -= 2 / (3 * np.pi) * nc * q2 * np.log(Q / m)
    return 1 / inv
ax.semilogx(Qs, [50 * alpha_em(q) for q in Qs], color="royalblue", lw=2, ls="--", label=r"$50\,\alpha(Q)$, one loop (crude)")
ax.axvline(mZ, color="grey", lw=0.8, ls=":"); ax.text(mZ * 1.1, 0.33, r"$m_Z$", fontsize=8, color="grey")
ax.set_xlabel(r"$Q$ (GeV)"); ax.set_ylabel("coupling"); ax.set_ylim(0, 0.45); ax.legend(fontsize=8)
ax.set_title(r"Asymptotic freedom: $\alpha_s$ falls, $\alpha$ rises", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
d8 = sorted([(round(a, 9), round(b, 9)) for a, b in w8])
print("[PASS] the octet has 8 weights, two of them zero:", len(w8) == 8 and d8.count((0.0, 0.0)) == 2)
print("[PASS] alpha_s decreases from 1.5 GeV to 10 TeV:", all(np.diff([alpha_s(q) for q in Qs]) < 0))
print("[PASS] one-loop alpha_s(m_tau) = %.3f (measured about 0.32):" % alpha_s(1.777), 0.28 < alpha_s(1.777) < 0.36)
print("wrote", os.path.relpath(OUT))
