# Exercise exr-jacobian-polar: the Jacobian matrix of the polar map, its determinant, and where the inverse function
# theorem applies (det != 0), with Sage's jacobian().
r, ph = var("r ph", domain="real"); assume(r > 0); assume(ph > -pi, ph < pi)
F = vector([r * cos(ph), r * sin(ph)])
J = jacobian(F, (r, ph))
print("Jacobian matrix:", J.rows())
print("det J =", J.det().simplify_full(), " -> nonzero exactly for r != 0")
print("inverse Jacobian at (r, ph) = (2, pi/3):", J.subs(r=2, ph=pi/3).inverse().simplify_full().rows())
x, y = var("x y", domain="real")
G = vector([sqrt(x ** 2 + y ** 2), atan2(y, x)])
JG = jacobian(G, (x, y))
print("chain rule: J_G(F(r,ph)) J_F(r,ph) = identity (for r > 0):", (JG.subs(x=F[0], y=F[1]) * J).simplify_full() == identity_matrix(2))
print("so J_G at F(r, ph) is the inverse Jacobian:", J.inverse().simplify_full().rows())
