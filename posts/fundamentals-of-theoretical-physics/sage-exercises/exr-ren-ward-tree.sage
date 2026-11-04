# Exercise exr-ren-ward-tree: the tree-level Ward-Takahashi identity S0(p') qslash S0(p) = S0(p) - S0(p') with the Weyl
# gamma matrices, S0(p) = (pslash - m)^-1, q = p' - p, at random momenta.
s = [matrix(CDF, [[0, 1], [1, 0]]), matrix(CDF, [[0, -I], [I, 0]]), matrix(CDF, [[1, 0], [0, -1]])]
one, Z = identity_matrix(CDF, 2), zero_matrix(CDF, 2)
g = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, sk], [-sk, Z]], subdivide=False) for sk in s]
eta = diagonal_matrix([1, -1, -1, -1])
slash = lambda p: sum(eta[mu, mu] * p[mu] * g[mu] for mu in range(4))
print("Clifford relations hold:", all((g[a] * g[b] + g[b] * g[a] - 2 * eta[a, b] * identity_matrix(4)).norm() < 1e-12 for a in range(4) for b in range(4)))
set_random_seed(7); m = 0.7
for trial in range(3):
    p = vector(RDF, [RDF.random_element() for _ in range(4)]); pp = vector(RDF, [RDF.random_element() for _ in range(4)])
    S0 = lambda k: (slash(k) - m * identity_matrix(CDF, 4)).inverse()
    lhs = S0(pp) * slash(pp - p) * S0(p); rhs = S0(p) - S0(pp)
    print("trial %d: |lhs - rhs| = %.2e" % (trial, (lhs - rhs).norm()))
