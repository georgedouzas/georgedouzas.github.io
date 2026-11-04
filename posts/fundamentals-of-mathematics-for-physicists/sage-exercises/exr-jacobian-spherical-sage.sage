# Exercise exr-jacobian-spherical-sage: the Jacobian matrix of spherical coordinates (r, theta, phi) -> (x, y, z), its
# determinant r^2 sin(theta), the points where it vanishes, and the inverse Jacobian as the inverse matrix.
r, th, ph = var("r theta phi")
F = vector([r * sin(th) * cos(ph), r * sin(th) * sin(ph), r * cos(th)])
J = jacobian(F, [r, th, ph])
print("J =", J.rows())
d = J.det().simplify_trig()
print("det J =", d, "; vanishes where r = 0 or sin(theta) = 0 (the poles):", solve(d == 0, [r, th])[:2])
Jinv = J.inverse().simplify_full()
print("J^-1 J = 1:", (Jinv * J).simplify_full() == identity_matrix(3))
print("first row of J^-1 (the gradient of r = sqrt(x^2+y^2+z^2) in spherical coordinates):", Jinv.row(0).simplify_trig())
