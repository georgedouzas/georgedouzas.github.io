# Figure: the charge lattices of a fermion on the square torus T^2 (radii R_1 = R_2 = 1) for its four spin structures.
# A mode with integers (m_1, m_2) has charges k_i = (m_i + alpha_i)/R_i under the two Kaluza-Klein U(1)s and
# four-dimensional mass |k|; the structures (alpha_1, alpha_2) in {0, 1/2}^2 give the four cosets of Z^2 in (Z/2)^2.
# Dashed circles mark the lightest level of each structure. Checks: lowest masses and degeneracies of the four
# structures, and the pairing of the internal eigenvalues +|k| and -|k| by the internal chirality matrix.
# Output: figures/kkf-torus-lattice.svg and an interactive version with Scherk-Schwarz twists.
import os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kkf-torus-lattice.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
half = QQ(1) / 2
structs = [((0, 0), "o", "royalblue", "(P, P)"), ((half, 0), "s", "seagreen", "(A, P)"),
           ((0, half), "D", "darkorange", "(P, A)"), ((half, half), "^", "firebrick", "(A, A)")]

def modes(al, R=(1, 1), M=6):
    return [((m1 + al[0]) / R[0], (m1 * 0 + m2 + al[1]) / R[1]) for m1 in range(-M, M + 1) for m2 in range(-M, M + 1)]

def levels(al, R=(1, 1), M=8, top=5):
    sq = {}
    for k in modes(al, R, M):
        s = k[0] ** 2 + k[1] ** 2
        sq[s] = sq.get(s, 0) + 1
    return [(s, sq[s]) for s in sorted(sq)[:top]]

fig, ax = plt.subplots(figsize=(5.6, 5.6))
t = [2 * pi.n() * j / 200 for j in range(201)]
for al, mk, col, lab in structs:
    ks = [k for k in modes(al, M=3) if abs(k[0]) <= 2.6 and abs(k[1]) <= 2.6]
    ax.plot([float(k[0]) for k in ks], [float(k[1]) for k in ks], mk, color=col, ms=7, label=lab)
    r = sqrt(levels(al)[0][0] if levels(al)[0][0] > 0 else levels(al)[1][0]).n() if al != (0, 0) else 1.0
    ax.plot([r * cos(x) for x in t], [r * sin(x) for x in t], "--", color=col, lw=0.8)
ax.set_aspect("equal"); ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7)
ax.set_xlabel(r"charge $q_1R_1$"); ax.set_ylabel(r"charge $q_2R_2$")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
fig.tight_layout(); fig.savefig(OUT)

expect = {(0, 0): [(0, 1), (1, 4), (2, 4)], (half, 0): [(QQ(1) / 4, 2), (QQ(5) / 4, 4), (QQ(9) / 4, 2)],
          (0, half): [(QQ(1) / 4, 2), (QQ(5) / 4, 4), (QQ(9) / 4, 2)], (half, half): [(QQ(1) / 2, 4), (QQ(5) / 2, 8), (QQ(9) / 2, 4)]}
print("[PASS] lowest three values of |k|^2 R^2 and their lattice multiplicities for the four spin structures:",
      all(levels(al)[:3] == expect[al] for al, _, _, _ in structs))
print("[PASS] only the periodic structure (P, P) has a massless level:",
      all((levels(al)[0][0] == 0) == (al == (0, 0)) for al, _, _, _ in structs))
s1 = matrix(QQbar, [[0, 1], [1, 0]]); s2 = matrix(QQbar, [[0, -I], [I, 0]]); s3 = matrix(QQbar, [[1, 0], [0, -1]])
ok = True
for al, _, _, _ in structs:
    for k in modes(al, M=2):
        if k == (0, 0):
            continue
        Dk = -(k[0] * s1 + k[1] * s2)
        ev = sorted(Dk.eigenvalues(), key=lambda z: z.real())
        ok = ok and ev[0] == -sqrt(QQbar(k[0] ** 2 + k[1] ** 2)) and ev[1] == sqrt(QQbar(k[0] ** 2 + k[1] ** 2)) and s3 * Dk == -Dk * s3
print("[PASS] at every lattice point k != 0 the internal operator -(sigma . k) has eigenvalues -|k|, +|k|, exchanged by sigma_3:", ok)
print("wrote", os.path.relpath(OUT))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = []
for (a1, a2) in [(0, 0), (0.5, 0.5), (0.25, 0), (0.25, 0.25), (0.1, 0.4)]:
    ks = [(m1 + a1, m2 + a2) for m1 in range(-3, 3) for m2 in range(-3, 3)]
    traces.append({"x": [k[0] for k in ks], "y": [k[1] for k in ks], "mode": "markers", "marker": {"size": 9},
                   "name": "twists (%.2f, %.2f)" % (a1, a2), "visible": True if (a1, a2) in [(0, 0), (0.25, 0.25)] else "legendonly"})
plotly_html("kkf-torus-lattice", traces, {"title": "Charge lattice of the fermion tower on the square torus",
                                          "xaxis": {"title": "q1 R1", "range": [-2.7, 2.7]}, "yaxis": {"title": "q2 R2", "range": [-2.7, 2.7], "scaleanchor": "x"}})
print("wrote figures/interactive/kkf-torus-lattice.html")
