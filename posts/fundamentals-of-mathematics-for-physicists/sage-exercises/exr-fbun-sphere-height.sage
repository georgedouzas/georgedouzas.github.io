# Exercise exr-fbun-sphere-height: the height z restricted to the sphere minus its poles is a trivial bundle over
# (-1, 1) with fibre the circle: the map (theta, phi) -> (z, phi) = (cos theta, phi) is a diffeomorphism onto
# (-1, 1) x S^1 compatible with the projections; on the whole sphere the fibres over z = +-1 are points.
S2 = Manifold(2, "S2"); U = S2.open_subset("U")
sph = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
B = Manifold(2, "B")                                         # (-1, 1) x S^1 with coordinates (z, phi)
cyl = B.chart(r"z:(-1,1) p:(0,2*pi):\varphi"); z, p = cyl[:]
Phi = U.diff_map(B, {(sph, cyl): [cos(th), ph]}, name="Phi")
J = Phi.jacobian_matrix(sph, cyl)
print("Jacobian of (theta, phi) -> (cos theta, phi):"); print(J)
print("determinant:", J.det(), "; nonzero for 0 < theta < pi:", bool(J.det().subs(th=pi/3) != 0))
print("inverse map z -> theta = arccos z is smooth on (-1, 1); derivative:", diff(arccos(z), z))
print("Phi followed by the projection (z, phi) -> z is the height cos(theta):", Phi.expr()[0])
print("fibre of the height over z = 1 on the full sphere: theta = 0, a single point; over z = 1/2:",
      solve(cos(th) == 1/2, th), "with phi arbitrary, a circle")
