# Exercise exr-vcp-hodge-monomials: counting complex-structure deformations. Homogeneous polynomials of degree d in
# n variables span a space of dimension binomial(n + d - 1, d); linear changes of the coordinates (GL(n), dimension n^2)
# do not change the hypersurface, so the number of deformations is binomial(n + d - 1, d) - n^2: 101 for the quintic
# (= h^{2,1}) and 19 for the quartic K3 surface (its algebraic deformations).
for name, n, d in [("quintic in CP^4", 5, 5), ("quartic in CP^3", 4, 4)]:
    P = PolynomialRing(QQ, n, "z")
    mons = [m for m in P.monomials_of_degree(d)]
    print(name, ": monomials of degree", d, "=", len(mons), "= binomial:", len(mons) == binomial(n + d - 1, d), "; deformations", len(mons), "-", n ** 2, "=", len(mons) - n ** 2)
