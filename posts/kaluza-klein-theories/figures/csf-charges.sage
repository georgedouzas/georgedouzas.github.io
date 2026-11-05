# Figure: charges of the components of the 3 and the 8 of SU(3) under the embedded isotropy U(1) of S^2 = SU(2)/U(1), for
# the lambda_8 embedding tau = (i/3) diag(1, 1, -2) of the running model and the T_3 embedding tau' = (i/2) diag(1, -1, 0).
# A component of charge q survives the fermion constraint when q + s = 0 for a spinor charge s = -+1/2, i.e. |q| = 1/2
# (dashed lines). Output: figures/csf-charges.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csf-charges.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
taus = {"lambda8": diagonal_matrix(QQ, [QQ(1) / 3, QQ(1) / 3, -QQ(2) / 3]), "T3": diagonal_matrix(QQ, [QQ(1) / 2, -QQ(1) / 2, 0])}
def charges(t):
    d = t.diagonal()
    fund = list(d)
    adj = [d[i] - d[j] for i in range(3) for j in range(3) if i != j] + [0, 0]
    return fund, adj
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, (name, title) in zip(axs, (("lambda8", r"$\tau = \frac{i}{3}\mathrm{diag}(1, 1, -2)$"), ("T3", r"$\tau' = \frac{i}{2}\mathrm{diag}(1, -1, 0)$"))):
    fund, adj = charges(taus[name])
    for x0, qs, lab in ((0, fund, r"$\mathbf{3}$"), (1, adj, r"$\mathbf{8}$")):
        from collections import Counter
        c = Counter(qs)
        for q, m in c.items():
            col = "firebrick" if abs(q) == QQ(1) / 2 else "royalblue"
            ax.plot([x0 + 0.08 * (k - (m - 1) / 2) for k in range(m)], [float(q)] * m, "o", color=col, ms=8)
        ax.text(x0, -1.45, lab, ha="center")
    for s in (0.5, -0.5):
        ax.axhline(s, ls="--", color="grey", lw=0.8)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-1.6, 1.3); ax.set_xticks([]); ax.set_title(title, fontsize=10)
axs[0].set_ylabel(r"charge $q$ under the isotropy $U(1)$")
fig.tight_layout(); fig.savefig(OUT)
f8, a8 = charges(taus["lambda8"]); f3, a3 = charges(taus["T3"])
print("[PASS] lambda_8 embedding: no component of the 3 or the 8 has |q| = 1/2:", all(abs(q) != QQ(1) / 2 for q in f8 + a8))
print("[PASS] lambda_8 embedding: the charges of the 3 have 2q not an integer (no bundle for the 3):", all((2 * q) not in ZZ for q in f8))
print("[PASS] T_3 embedding: the 3 has charges 1/2, -1/2, 0 and the 8 has four components with |q| = 1/2:",
      sorted(f3) == [-QQ(1) / 2, 0, QQ(1) / 2] and len([q for q in a3 if abs(q) == QQ(1) / 2]) == 4)
print("[PASS] the charges sum to zero in each representation (traceless tau):", sum(f8) == 0 and sum(a8) == 0 and sum(f3) == 0 and sum(a3) == 0)
print("wrote", os.path.relpath(OUT))
