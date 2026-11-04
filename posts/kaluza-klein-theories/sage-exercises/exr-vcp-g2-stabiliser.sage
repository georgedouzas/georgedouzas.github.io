# Exercise exr-vcp-g2-stabiliser: G_2 as the stabiliser of the associative 3-form Phi = sum over the seven triples
# (123), (147), (165), (246), (257), (354), (367) of e^{ijk} (Acharya-Gukov, eq. 2.8). The infinitesimal stabiliser
# {X in gl(7) : X . Phi = 0} is 14-dimensional and consists of antisymmetric matrices: g_2 inside so(7). The orbit
# of Phi has dimension 49 - 14 = 35 = dim Lambda^3(R^7): Phi is a generic (open-orbit) 3-form.
triples = [(1, 2, 3), (1, 4, 7), (1, 6, 5), (2, 4, 6), (2, 5, 7), (3, 5, 4), (3, 6, 7)]
Phi = {}
for (i, j, k) in triples:
    i, j, k = i - 1, j - 1, k - 1
    for (p, sgn) in [((i, j, k), 1), ((j, k, i), 1), ((k, i, j), 1), ((j, i, k), -1), ((i, k, j), -1), ((k, j, i), -1)]:
        Phi[p] = sgn
def phi(i, j, k): return Phi.get((i, j, k), 0)
xs = var(["x%d_%d" % (a, b) for a in range(7) for b in range(7)])
X = matrix(SR, 7, 7, xs)
eqs = []
for i in range(7):
    for j in range(i + 1, 7):
        for k in range(j + 1, 7):
            eqs.append(sum(X[l, i] * phi(l, j, k) + X[l, j] * phi(i, l, k) + X[l, k] * phi(i, j, l) for l in range(7)))
A = matrix(QQ, [[QQ(e.coefficient(x)) for x in xs] for e in eqs])
ker = A.right_kernel()
print("number of linear conditions:", A.nrows(), "; rank:", A.rank(), "; dim of stabiliser:", ker.dimension())
antisym = all(matrix(QQ, 7, 7, list(v)).transpose() == -matrix(QQ, 7, 7, list(v)) for v in ker.basis())
print("every element of the stabiliser is antisymmetric (g_2 in so(7)):", antisym)
print("orbit dimension 49 - 14 =", 49 - ker.dimension(), "= dim Lambda^3 R^7 =", binomial(7, 3))
