# Exercise exr-polar-metric-spherical: the Euclidean metric of R^3 in spherical and in cylindrical coordinates, pulled
# back through the coordinate maps with SageManifolds; matrices, determinants and inverses extend the polar metric.
E3 = Manifold(3, "R^3", structure="Riemannian"); cart = E3.chart("x y z")
delta = E3.metric("delta"); delta[0, 0] = 1; delta[1, 1] = 1; delta[2, 2] = 1
S = Manifold(3, "S", structure="Riemannian"); sph = S.chart(r"rho:(0,+oo):\rho th:(0,pi):\theta ph:(0,2*pi):\varphi"); rho, th, ph = sph[:]
Phi = S.diff_map(E3, {(sph, cart): [rho*sin(th)*cos(ph), rho*sin(th)*sin(ph), rho*cos(th)]})
g = S.metric("g"); g.set(Phi.pullback(delta))
print("spherical coordinates:", g.display())
print("matrix:"); print(g[:])
print("determinant:", g.determinant().expr().simplify_full(), " -> volume element rho^2 sin(theta)")
print("inverse metric:", g.inverse().display())
C = Manifold(3, "C", structure="Riemannian"); cyl = C.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi z"); r, phc, z = cyl[:]
Psi = C.diff_map(E3, {(cyl, cart): [r*cos(phc), r*sin(phc), z]})
gc = C.metric("g"); gc.set(Psi.pullback(delta))
print("cylindrical coordinates:", gc.display(), "; det =", gc.determinant().expr().simplify_full(), "; inverse:", gc.inverse().display())
