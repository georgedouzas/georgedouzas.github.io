# Exercise exr-inverse-function-sage: the map (x, y) -> (u, v) = (x^2 - y^2, 2xy) (the complex squaring z -> z^2) has
# Jacobian determinant 4(x^2 + y^2), non-zero away from the origin, so it is a local diffeomorphism there but not
# injective (it is 2-to-1); the Jacobian of the local inverse is the inverse matrix.
x, y = var("x y")
F = vector([x ** 2 - y ** 2, 2 * x * y])
J = jacobian(F, [x, y])
print("J =", J.rows(), "; det J =", J.det().factor())
print("F(1, 1) =", F.subs(x=1, y=1), "= F(-1, -1) =", F.subs(x=-1, y=-1), ": not injective, each value is taken twice")
Jinv = J.inverse()
print("Jacobian of the local inverse at (1, 1):", Jinv.subs(x=1, y=1), "; product with J:", (Jinv * J).simplify_full() == identity_matrix(2))
# the explicit local inverse near (1, 1): z = sqrt(u + i v), checked by composition
u, v = var("u v")
print("check: F(x(u, v), y(u, v)) for the branch x = sqrt((sqrt(u^2+v^2) + u)/2), y = v/(2x) at (u, v) = (0, 2):", F.subs(x=sqrt((sqrt(4) + 0) / 2), y=2 / (2 * sqrt((sqrt(4) + 0) / 2))))
