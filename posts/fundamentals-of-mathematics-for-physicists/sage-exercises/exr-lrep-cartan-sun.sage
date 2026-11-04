# Exercise exr-lrep-cartan-sun: the Cartan subalgebra of su(n) is the set of diagonal traceless matrices; the
# centraliser of a generic diagonal element is exactly this set, so the rank of su(n) is n - 1.
for n in [2, 3, 4, 5]:
    R = PolynomialRing(QQ, ["x%d_%d" % (i, j) for i in range(n) for j in range(n)])
    X = matrix(R, n, n, R.gens())
    H = diagonal_matrix(QQ, [k + 1 for k in range(n - 1)] + [-sum(range(1, n))])   # distinct eigenvalues, traceless
    eqs = (H * X - X * H).list() + [X.trace()]
    M = matrix(QQ, [[e.monomial_coefficient(g) for g in R.gens()] for e in eqs])
    dim_centraliser = n * n - M.rank()
    D = [diagonal_matrix(QQ, [1 if k == i else (-1 if k == i + 1 else 0) for k in range(n)]) for i in range(n - 1)]
    commute = all(A * B == B * A for A in D for B in D)
    print(f"su({n}): H = diag{tuple(H.diagonal())}; traceless centraliser has dimension {dim_centraliser} = n - 1: {dim_centraliser == n - 1};",
          "diagonal traceless basis commutes:", commute)
