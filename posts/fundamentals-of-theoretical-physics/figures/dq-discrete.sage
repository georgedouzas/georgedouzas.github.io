# Figure: the action of charge conjugation C, parity P and time reversal T on a spin-one-half particle of charge q,
# momentum p (long arrow) and spin s (short double arrow) along the momentum: C flips the charge only, P flips the
# momentum but not the spin (so the helicity), T flips both (so not the helicity). The checks are the matrix identities
# behind psi^c = i gamma^2 psi^*, psi_P = gamma^0 psi(t, -x), psi_T = i gamma^1 gamma^3 psi^*(-t, x) in the Weyl
# representation. Output: figures/dq-discrete.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-discrete.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(9.5, 2.4))
cases = [("original", "q", 1, 1), ("C", "-q", 1, 1), ("P", "q", -1, 1), ("T", "q", -1, -1)]
for ax, (ttl, ch, pd, sd) in zip(axs, cases):
    ax.add_patch(plt.Circle((0, 0), 0.32, color="lightsteelblue"))
    ax.text(0, 0, r"$%s$" % ch, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(1.3 * pd, 0), xytext=(0.4 * pd, 0), arrowprops=dict(arrowstyle="->", lw=2, color="firebrick"))
    ax.text(0.85 * pd, 0.13, r"$\vec p$", color="firebrick", ha="center")
    ax.annotate("", xy=(0.55 * sd, -0.55), xytext=(-0.55 * sd, -0.55), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="darkgreen"))
    ax.text(0, -0.8, r"$\vec s$", color="darkgreen", ha="center")
    hel = "+" if pd * sd > 0 else "-"
    ax.set_title(r"%s: helicity $%s$" % (ttl, hel), fontsize=9)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.1, 0.6); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT)
s0 = identity_matrix(QQbar, 2); s1 = matrix(QQbar, [[0,1],[1,0]]); s2 = matrix(QQbar, [[0,-I],[I,0]]); s3 = matrix(QQbar, [[1,0],[0,-1]]); Z = matrix(QQbar, 2, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, s0, s0, Z)] + [blk(Z, s, -s, Z) for s in (s1, s2, s3)]
g5 = I * g[0] * g[1] * g[2] * g[3]
C = I * g[2] * g[0]; A = I * g[2]; T = I * g[1] * g[3]
print("[PASS] C = i gamma^2 gamma^0 satisfies C gamma^mu C^-1 = -(gamma^mu)^T and C^T = -C:", all(C * g[k] * C.inverse() == -g[k].transpose() for k in range(4)) and C.transpose() == -C)
print("[PASS] (i gamma^2)(gamma^mu)^*(i gamma^2)^-1 = -gamma^mu, so psi^c = i gamma^2 psi^* solves the equation with q -> -q:", all(A * g[k].conjugate() * A.inverse() == -g[k] for k in range(4)))
print("[PASS] (psi^c)^c = psi, i.e. (i gamma^2)(i gamma^2)^* = 1:", A * A.conjugate() == identity_matrix(QQbar, 4))
print("[PASS] gamma^0 gamma^0 gamma^0 = gamma^0 and gamma^0 gamma^k gamma^0 = -gamma^k (parity):", all(g[0] * g[k] * g[0] == (1 if k == 0 else -1) * g[k] for k in range(4)))
print("[PASS] T (gamma^0)^* T^-1 = gamma^0, T (gamma^k)^* T^-1 = -gamma^k and T T^* = -1 (Kramers):", T * g[0].conjugate() * T.inverse() == g[0] and all(T * g[k].conjugate() * T.inverse() == -g[k] for k in (1, 2, 3)) and T * T.conjugate() == -identity_matrix(QQbar, 4))
M = A * g[0].conjugate() * T.conjugate()
print("[PASS] the combined CPT matrix is -i gamma_5:", M == -I * g5)
print("wrote", os.path.relpath(OUT))
