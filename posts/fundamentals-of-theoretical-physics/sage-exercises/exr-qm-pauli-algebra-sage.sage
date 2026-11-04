# Exercise exr-qm-pauli-algebra-sage: the Pauli product rule, (a.sigma)(b.sigma) = a.b + i (a x b).sigma, traces, determinants.
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
one = identity_matrix(SR, 2)
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1])) if len({i, j, k}) == 3 else 0
print("sigma_i sigma_j = delta_ij + i eps_ijk sigma_k for all i, j:",
      all(sig[i] * sig[j] == (1 if i == j else 0) * one + I * sum(eps(i, j, k) * sig[k] for k in range(3)) for i in range(3) for j in range(3)))
a = vector(SR, var("a1 a2 a3")); b = vector(SR, var("b1 b2 b3"))
dot = lambda v: sum(v[i] * sig[i] for i in range(3))
lhs = dot(a) * dot(b); rhs = (a * b) * one + I * dot(a.cross_product(b))
print("(a.sigma)(b.sigma) - a.b - i (a x b).sigma =", (lhs - rhs).simplify_full().list())
print("traces:", [s.trace() for s in sig], " determinants:", [s.det() for s in sig])
