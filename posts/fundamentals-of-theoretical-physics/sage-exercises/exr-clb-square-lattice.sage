# Exercise exr-clb-square-lattice (Foulaadvand, Problem 4.12): small transverse oscillations of an N x N square lattice of
# equal masses joined by equal springs, fixed at all four sides (m = k = 1, one displacement component per mass). The
# stiffness matrix is the Kronecker sum of two chains, so omega^2 = 4 (sin^2(p pi/(2(N+1))) + sin^2(q pi/(2(N+1)))).
N = 4
C = matrix(RDF, N, N, lambda i, j: 2 if i == j else (-1 if abs(i - j) == 1 else 0))
I = identity_matrix(RDF, N)
K = C.tensor_product(I) + I.tensor_product(C)
w2 = sorted(K.eigenvalues())
pred = sorted([4 * (sin(p * pi / (2 * (N + 1)))^2 + sin(q * pi / (2 * (N + 1)))^2).n() for p in range(1, N + 1) for q in range(1, N + 1)])
print("all %d frequencies match the formula:" % (N * N), all(abs(u - v) < 1e-10 for u, v in zip(w2, pred)))
print("lowest omega = %.5f (p = q = 1), highest = %.5f (p = q = N)" % (sqrt(w2[0]), sqrt(w2[-1])))
from collections import Counter
cnt = Counter(round(v, 8) for v in w2)
print("degenerate levels (omega^2: multiplicity):", sorted((k, m) for k, m in cnt.items() if m > 1))
