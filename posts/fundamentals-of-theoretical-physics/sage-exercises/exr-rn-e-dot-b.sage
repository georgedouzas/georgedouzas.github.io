# Exercise exr-rn-e-dot-b: with F_{0i} = E_i, F_{ij} = -eps_{ijk} B_k (lower indices, signature (+,-,-,-)) and
# eps^{0123} = 1: eps^{mu nu rho sigma} F_{mu nu} F_{rho sigma} = -8 E.B and F_{mu nu} F^{mu nu} = 2(B^2 - E^2).
E = vector(var("E1 E2 E3")); B = vector(var("B1 B2 B3"))
F = matrix(SR, 4, 4)
for i in range(3):
    F[0, i + 1] = E[i]; F[i + 1, 0] = -E[i]
    for j in range(3):
        F[i + 1, j + 1] = -sum((QQ(1) if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else (-1 if (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)] else 0)) * B[k] for k in range(3))
from itertools import permutations
def eps4(idx):
    if len(set(idx)) < 4:
        return 0
    return Permutation([i + 1 for i in idx]).sign()
eFF = sum(eps4(p) * F[p[0], p[1]] * F[p[2], p[3]] for p in permutations(range(4)))
eta = diagonal_matrix([1, -1, -1, -1])
Fup = eta * F * eta
FF = sum(F[m, n] * Fup[m, n] for m in range(4) for n in range(4))
print("eps F F =", eFF.expand(), "  = -8 E.B:", bool(eFF == -8 * E.dot_product(B)))
print("F F =", FF.expand(), "  = 2(B^2 - E^2):", bool(FF == 2 * (B.dot_product(B) - E.dot_product(E))))
