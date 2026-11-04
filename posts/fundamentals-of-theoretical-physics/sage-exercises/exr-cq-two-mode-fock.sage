# Exercise exr-cq-two-mode-fock: two modes (omega_1 = 1, omega_2 = 2) truncated to four levels each, as a tensor
# product: commutators, [H, N] = 0, and the energies of H with their degeneracies.
M = 4
a = matrix(QQbar, M, M, lambda i, j: sqrt(j) if j == i + 1 else 0); one = identity_matrix(QQbar, M)
a1 = a.tensor_product(one); a2 = one.tensor_product(a)
d = lambda X: X.conjugate_transpose()
print("[a1, a2^dag] = 0:", a1 * d(a2) - d(a2) * a1 == 0)
H = 1 * d(a1) * a1 + 2 * d(a2) * a2; Nop = d(a1) * a1 + d(a2) * a2
print("[H, N] = 0:", H * Nop - Nop * H == 0, "  H diagonal:", H.is_diagonal())
from collections import Counter
c = Counter(ZZ(e) for e in H.diagonal())
print("energy: degeneracy  ", sorted(c.items()))
print("states with E = 2: (n1, n2) =", [(i // M, i % M) for i in range(M * M) if H[i, i] == 2])
