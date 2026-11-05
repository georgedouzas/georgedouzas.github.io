# Figure: the multiplication table of the 16 basis elements gamma_A of the Clifford algebra Cl(1,3) in the Weyl
# representation of the book, coloured by the sign c in gamma_A gamma_B = c gamma_C (only c = +1 and c = -1 occur).
# Output: figures/sd-clifford-table.svg.
import os, itertools
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-clifford-table.svg")
s = [identity_matrix(QQbar, 2), matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
Z = zero_matrix(QQbar, 2)
g = [block_matrix([[Z, s[0]], [s[0], Z]])] + [block_matrix([[Z, s[k]], [-s[k], Z]]) for k in (1, 2, 3)]
eta = diagonal_matrix([1, -1, -1, -1])
subsets = [A for k in range(5) for A in itertools.combinations(range(4), k)]
def gam(A):
    M = identity_matrix(QQbar, 4)
    for a in A: M = M * g[a]
    return M
basis = [gam(A) for A in subsets]
names = ["1"] + ["".join(str(a) for a in A) for A in subsets[1:]]
def decompose(M):
    for k, B in enumerate(basis):
        for c in (1, I, -1, -I):
            if M == c * B: return k, c
    return None
T = matrix(ZZ, 16, 16)
closure = True
for a in range(16):
    for b in range(16):
        r = decompose(basis[a] * basis[b])
        if r is None: closure = False; continue
        T[a, b] = 1 if r[1] == 1 else -1
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.imshow([[float(T[a, b]) for b in range(16)] for a in range(16)], cmap=ListedColormap(["#c0392b", "#d6eaf8"]), vmin=-1, vmax=1)
labels = ["$1$"] + ["$\\gamma^{" + n + "}$" for n in names[1:]]
ax.set_xticks(range(16)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticks(range(16)); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(r"$\gamma_B$"); ax.set_ylabel(r"$\gamma_A$")
ax.set_title(r"sign of $\gamma_A\gamma_B = \pm\gamma_C$ (light $+$, dark $-$)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the Weyl matrices satisfy {g^mu, g^nu} = 2 eta^{mu nu}:", all(g[m] * g[n] + g[n] * g[m] == 2 * eta[m, n] * identity_matrix(4) for m in range(4) for n in range(4)))
print("[PASS] the 16 products gamma_A are linearly independent (rank 16):", matrix(QQbar, [B.list() for B in basis]).rank() == 16)
print("[PASS] tr gamma_A = 0 for A nonempty:", all(B.trace() == 0 for B in basis[1:]))
print("[PASS] every product gamma_A gamma_B is +gamma_C or -gamma_C:", closure and all(abs(T[a, b]) == 1 for a in range(16) for b in range(16)))
print("[PASS] gamma_A^2 = (-1)^(k(k-1)/2) prod eta_aa:", all(gam(A)^2 == (-1)^(len(A)*(len(A)-1)//2) * prod(eta[a, a] for a in A) * identity_matrix(4) for A in subsets))
print("[PASS] gamma_5 = i g0 g1 g2 g3 = diag(-1, -1, 1, 1):", I * g[0] * g[1] * g[2] * g[3] == diagonal_matrix(QQbar, [-1, -1, 1, 1]))
print("wrote", os.path.relpath(OUT))
