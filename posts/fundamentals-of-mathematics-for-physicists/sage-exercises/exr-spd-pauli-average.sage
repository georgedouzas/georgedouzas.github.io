# Exercise exr-spd-pauli-average: Pauli's averaging formula T = sum_A gamma_A F gamma'_A^{-1} intertwines the standard
# four-dimensional gamma matrices of signature (1,3) with the Weyl matrices of the book.
import itertools
s = [identity_matrix(QQbar, 2), matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
kr = lambda a, b: a.tensor_product(b)
standard = [kr(s[1], s[0]), I * kr(s[2], s[0]), I * kr(s[3], s[1]), I * kr(s[3], s[2])]
weyl = [kr(s[1], s[0])] + [kr(I * s[2], s[k]) for k in (1, 2, 3)]
eta = diagonal_matrix([1, -1, -1, -1])
for G in (standard, weyl):
    print("Clifford relations with eta = diag(1,-1,-1,-1):", all(G[a] * G[b] + G[b] * G[a] == 2 * eta[a, b] for a in range(4) for b in range(4)))
subsets = [A for k in range(5) for A in itertools.combinations(range(4), k)]
P = lambda G, A: prod([G[a] for a in A], identity_matrix(QQbar, 4))
set_random_seed(3)
F = random_matrix(QQ, 4, 4, num_bound=3, den_bound=1).change_ring(QQbar)
T = sum(P(standard, A) * F * P(weyl, A).inverse() for A in subsets)
print("det T =", T.det())
print("gamma^a T = T gamma'^a for a = 0..3:", all(standard[a] * T == T * weyl[a] for a in range(4)))
