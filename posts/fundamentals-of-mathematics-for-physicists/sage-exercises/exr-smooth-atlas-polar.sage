# Exercise exr-smooth-atlas-polar: Cartesian coordinates (x, y) and polar coordinates (r, phi) on the plane minus the
# half-line y = 0, x >= 0 form a smooth atlas: the transition map (r, phi) -> (r cos phi, r sin phi) has Jacobian
# determinant r > 0 and the smooth inverse r = sqrt(x^2 + y^2), phi = atan2(-y, -x) + pi in (0, 2 pi).
P = Manifold(2, "P")                                    # the plane minus the half-line y = 0, x >= 0
cart = P.chart("x y"); pol = P.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi")
x, y = cart[:]; r, ph = pol[:]
tr = pol.transition_map(cart, (r*cos(ph), r*sin(ph)))
tr.set_inverse(sqrt(x**2 + y**2), atan2(-y, -x) + pi, check=False)
print("polar -> Cartesian:", tr.display())
print("Cartesian -> polar:", tr.inverse().display())
J = jacobian(tr(r, ph), (r, ph))
print("Jacobian matrix:", J)
print("Jacobian determinant:", J.det().simplify_full(), " (SageManifolds:", tr.jacobian_det().expr().simplify_full(), ")")
print("non-zero on the domain r > 0: the two charts are compatible")
print("check of the inverse at (x, y) = (-1, -1): (r, phi) =", (sqrt(2), 5*pi/4), "and back:", (sqrt(2)*cos(5*pi/4), sqrt(2)*sin(5*pi/4)))
print("the inverse formula gives phi = atan2(1, 1) + pi =", atan2(1, 1) + pi)
print("atlas of P:", P.atlas())
