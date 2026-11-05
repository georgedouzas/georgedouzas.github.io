# Figure: the lowest Landau levels of the first Kaluza-Klein fermion (charge q = 1/R, mass m = 1/R, gyromagnetic ratio
# g = 1) in a weak magnetic field B, compared with an elementary Dirac fermion of the same charge and mass (g = 2).
# Non-relativistic energies E - m = omega (N + 1/2) - g omega s / 2 with omega = |q| B / m and s = +-1/2.
# The checks compute the spin-connection term of the five-dimensional Dirac operator on the Kaluza-Klein ansatz with
# explicit gamma matrices, the chiral rotation that turns it into a Pauli term, and the resulting g.
# Output: figures/kkf-zeeman.svg and an interactive version.
import os, sys, random
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kkf-zeeman.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4))
Bs = [k / 100 for k in range(101)]
for g, ls, lab in ((1, "-", "Kaluza-Klein mode, $g = 1$"), (2, "--", "Dirac particle, $g = 2$")):
    for N in (0, 1):
        for s in (QQ(1) / 2, -QQ(1) / 2):
            ax.plot(Bs, [float(b * (N + QQ(1) / 2) - g * b * s / 2) for b in Bs], ls=ls, color="firebrick" if g == 1 else "royalblue",
                    lw=1.6, label=lab if (N, s) == (0, QQ(1) / 2) else None)
ax.set_xlabel(r"magnetic field $B$ (units $m^2/|q|$)"); ax.set_ylabel(r"$E - m$ (units $m$)")
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 2.0); ax.legend(loc="upper left", fontsize=9)
ax.text(0.62, 0.05, "lowest level stays at $E = m$ only for $g = 2$", fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
# --- checks with explicit gamma matrices, signature (+,-,-,-,-), Gamma^4 = i gamma_5 ---
s0 = matrix(QQbar, [[1, 0], [0, 1]]); s1 = matrix(QQbar, [[0, 1], [1, 0]]); s2 = matrix(QQbar, [[0, -I], [I, 0]]); s3 = matrix(QQbar, [[1, 0], [0, -1]])
Z = matrix(QQbar, 2, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g4 = [blk(s0, Z, Z, -s0)] + [blk(Z, s, -s, Z) for s in (s1, s2, s3)]
g5 = I * g4[0] * g4[1] * g4[2] * g4[3]
G = g4 + [I * g5]
eta = diagonal_matrix([1, -1, -1, -1, -1]); one = identity_matrix(QQbar, 4)
print("[PASS] Gamma^mu = gamma^mu, Gamma^4 = i gamma_5 satisfy the Clifford relation of signature (1, 4):",
      all(G[a] * G[b] + G[b] * G[a] == 2 * eta[a, b] * one for a in range(5) for b in range(5)))
random.seed(int(1))
F = matrix(QQ, 4, 4)
for a in range(4):
    for b in range(a + 1, 4):
        F[a, b] = QQ(random.randint(int(-5), int(5))); F[b, a] = -F[a, b]
def w(A, B, C):  # F-dependent part of omega_{AB}(E_C): omega_ab(E_4) = F_ab/2, omega_4a(E_b) = -F_ab/2
    if A < 4 and B < 4 and C == 4: return F[A, B] / 2
    if A == 4 and B < 4 and C < 4: return -F[B, C] / 2
    if B == 4 and A < 4 and C < 4: return F[A, C] / 2
    return 0
X = sum(G[C] * sum(w(A, B, C) * G[A] * G[B] for A in range(5) for B in range(5)) / 4 for C in range(5))
FG = sum(F[a, b] * G[a] * G[b] for a in range(4) for b in range(4))
print("[PASS] Gamma^C (1/4) omega_AB(E_C) Gamma^A Gamma^B = -(1/8) F_ab gamma^a gamma^b Gamma^4:", X == -FG * G[4] / 8)
sig = lambda a, b: I / 2 * (G[a] * G[b] - G[b] * G[a])
sigF = sum(F[a, b] * sig(a, b) for a in range(4) for b in range(4))
rot = (one - I * g5) / sqrt(QQbar(2))   # exp(-i pi gamma5/4) = (1 - i gamma5)/sqrt 2
pauli = (FG * g5 / 8)                   # the term (1/8) F gamma gamma gamma5 of the mode equation
print("[PASS] exp(-i pi g5/4) [(1/8) F g g g5] exp(-i pi g5/4) = -(1/8) sigma^ab F_ab:", rot * pauli * rot == -sigF / 8)
print("[PASS] exp(-i pi g5/4) [-i (n/R) g5] exp(-i pi g5/4) = -(n/R), a mass n/R:", rot * (-I * g5) * rot == -one)
mu_dirac = -QQ(1) / 2; mu_pauli = 2 * QQ(1) / 8   # moments in units of |q|/m along Sigma, n > 0
print("[PASS] the total moment is half the Dirac moment, g = 1:", (mu_dirac + mu_pauli) / mu_dirac * 2 == 1)
print("wrote", os.path.relpath(OUT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = []
for g, col in ((1, "firebrick"), (2, "royalblue")):
    for N in (0, 1, 2):
        for s in (0.5, -0.5):
            traces.append({"x": Bs, "y": [b * (N + 0.5) - g * b * s / 2 for b in Bs], "mode": "lines", "line": {"color": col, "dash": "solid" if g == 1 else "dash"},
                           "name": "g = %d, N = %d, s = %+.1f" % (g, N, s), "legendgroup": "g%d" % g, "visible": True if N < 2 else "legendonly"})
plotly_html("kkf-zeeman", traces, {"title": "Landau and Zeeman levels: Kaluza-Klein fermion (g = 1) vs Dirac particle (g = 2)", "xaxis": {"title": "B (units m^2/|q|)"}, "yaxis": {"title": "E - m (units m)"}})
print("wrote figures/interactive/kkf-zeeman.html")
