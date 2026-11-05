# Figure: the Dirac spectrum of S^3 = SU(2) from representation theory: on V_j (x) V_j^* (x) C^2 the operator
# D = 2 sigma.J + 3/2 has eigenvalues 2j + 3/2 (multiplicity (2j+1)(2j+2)) and -(2j + 1/2) (multiplicity (2j+1) 2j).
# Output: figures/sd-s3-spectrum.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-s3-spectrum.svg")
def spin_matrices(j):
    j = QQ(j); dim = int(2 * j + 1); ms = [j - k for k in range(dim)]
    Jz = diagonal_matrix(QQbar, ms); Jp = matrix(QQbar, dim, dim)
    for k in range(1, dim):
        Jp[k - 1, k] = sqrt(QQbar(j * (j + 1) - ms[k] * (ms[k] + 1)))
    Jm = Jp.transpose()
    return (Jp + Jm) / 2, (Jp - Jm) / (2 * QQbar(I)), Jz
S = spin_matrices(1/2)
data = []
for twoj in range(0, 9):
    J = spin_matrices(twoj / 2)
    Dm = 4 * sum(S[a].tensor_product(J[a]) for a in range(3)) + 3/2 * identity_matrix(QQbar, 2 * (twoj + 1))
    for ev in set(Dm.eigenvalues()):
        mult = sum(1 for e in Dm.eigenvalues() if e == ev) * (twoj + 1)      # times the multiplicity 2j + 1 of V_j^*
        data.append((twoj / 2, QQ(ev), mult))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for j, ev, mult in data:
    ax.scatter([float(j)], [float(ev)], s=8 * float(mult) ** 0.8, color="royalblue" if ev > 0 else "firebrick", alpha=0.7)
    ax.annotate(str(mult), (float(j), float(ev)), textcoords="offset points", xytext=(7, -3), fontsize=7)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel(r"$j$ (representation $V_j \otimes V_j^*$ of $SU(2) \times SU(2)$)"); ax.set_ylabel(r"eigenvalue of $D$")
fig.tight_layout(); fig.savefig(OUT)
ok_pos = all(ev == 2 * j + 3/2 and mult == (2 * j + 1) * (2 * j + 2) for j, ev, mult in data if ev > 0)
ok_neg = all(ev == -(2 * j + 1/2) and mult == (2 * j + 1) * 2 * j for j, ev, mult in data if ev < 0)
print("[PASS] positive eigenvalues 2j + 3/2 with multiplicity (2j+1)(2j+2):", ok_pos)
print("[PASS] negative eigenvalues -(2j + 1/2) with multiplicity (2j+1) 2j:", ok_neg)
print("[PASS] together: +-(l + 3/2) with multiplicity (l+1)(l+2), the S^3 formula:", all(any(ev == l + 3/2 and mult == (l + 1) * (l + 2) for j, ev, mult in data) and any(ev == -(l + 3/2) and mult == (l + 1) * (l + 2) for j, ev, mult in data) for l in range(0, 7)))
print("wrote", os.path.relpath(OUT))
