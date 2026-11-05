# Figure: relativistic Landau levels of a Dirac particle of charge q in a uniform field B along z (p_z = 0, m = 1):
# E_N = sqrt(m^2 + 2N|qB|), N = 0, 1, 2, ..., against the field in units of the critical field m^2/|q|, compared with
# the Pauli (g = 2) levels m + N|qB|/m. Output: figures/dq-landau.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-landau.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
Bs = np.linspace(0, 1.5, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
cols = ["black", "royalblue", "darkgreen", "darkorange", "firebrick"]
for N, c in enumerate(cols):
    ax.plot(Bs, np.sqrt(1 + 2 * N * Bs), color=c, lw=2, label=r"$N = %d$" % N)
    ax.plot(Bs, 1 + N * Bs, color=c, lw=1, ls="--")
ax.set_xlabel(r"$|qB|/m^2$"); ax.set_ylabel(r"$E_N/m$ (solid: Dirac; dashed: Pauli with $g = 2$)")
ax.set_ylim(0.9, 4); ax.legend(fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(OUT)
# checks: (sigma . pi)^2 = pi^2 - q B sigma_3 on two-component functions, for A = (-By/2, Bx/2, 0)
x, y, z, q, B, mm = var("x y z q B mm")
f1 = function("f1")(x, y, z); f2 = function("f2")(x, y, z)
A = [-B * y / 2, B * x / 2, 0]; X = [x, y, z]
pi = lambda k, h: -I * diff(h, X[k]) - q * A[k] * h
s = [matrix(SR, [[0,1],[1,0]]), matrix(SR, [[0,-I],[I,0]]), matrix(SR, [[1,0],[0,-1]])]
def sigpi(F):
    return vector(SR, [sum(s[k][r, c] * pi(k, F[c]) for k in range(3) for c in range(2)) for r in range(2)])
F = vector(SR, [f1, f2])
lhs = sigpi(sigpi(F))
pi2 = vector(SR, [sum(pi(k, pi(k, F[r])) for k in range(3)) for r in range(2)])
rhs = pi2 - q * B * (s[2] * F)
print("[PASS] (sigma . pi)^2 = pi^2 - qB sigma_3 for the symmetric gauge:", all(bool((a - b).expand() == 0) for a, b in zip(lhs, rhs)))
# the Dirac levels expand to the Pauli levels plus the relativistic correction -(N qB)^2/(2 m^3)
N, b = var("N b", domain="positive")
print("[PASS] sqrt(m^2 + 2N b) = m + N b/m - N^2 b^2/(2 m^3) + O(b^3):", bool(taylor(sqrt(mm^2 + 2 * N * b), b, 0, 2) == mm + N * b / mm - N^2 * b^2 / (2 * mm^3)))
print("[PASS] the second-order term is -p^4/(8 m^3) at p^2 = 2 N b:", bool(-(2 * N * b)^2 / (8 * mm^3) == -N^2 * b^2 / (2 * mm^3)))
print("wrote", os.path.relpath(OUT))
