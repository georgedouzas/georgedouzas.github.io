# Exercise exr-lgt-link-hopping: the hopping term phi(x)^dagger U phi(x + a mu) is invariant under a lattice gauge transformation.
set_random_seed(1)
def su2():                                   # a random SU(2) matrix a0 + i a.sigma
    v = vector(RDF, [RDF.random_element(-1, 1) for _ in range(4)]); v = v / v.norm()
    return matrix(CDF, [[v[0] + I*v[3], v[2] + I*v[1]], [-v[2] + I*v[1], v[0] - I*v[3]]])
U, gx, gy = su2(), su2(), su2()
phx = vector(CDF, [CDF.random_element() for _ in range(2)]); phy = vector(CDF, [CDF.random_element() for _ in range(2)])
before = phx.conjugate() * U * phy
U2 = gx.inverse() * U * gy; phx2 = gx.inverse() * phx; phy2 = gy.inverse() * phy
after = phx2.conjugate() * U2 * phy2
print('det U =', U.det().real().n(digits=12), '; U unitary:', (U * U.conjugate_transpose() - 1).norm() < 1e-12)
print('hopping term before:', before.n(digits=10))
print('hopping term after: ', after.n(digits=10))
print('difference:', abs(before - after) < 1e-12)
