# Exercise exr-change-variables-sage: the area of the ellipse x^2/a^2 + y^2/b^2 <= 1 by the change of variables
# x = a u, y = b v (Jacobian ab) and by polar-like coordinates x = a r cos t, y = b r sin t (Jacobian ab r); and the
# volume of the unit ball by spherical coordinates (Jacobian r^2 sin theta).
a, b, u, v, r, t, th, ph = var("a b u v r t theta phi")
assume(a > 0, b > 0)
J1 = jacobian([a * u, b * v], [u, v]).det()
J2 = jacobian([a * r * cos(t), b * r * sin(t)], [r, t]).det().simplify_trig()
print("Jacobians: linear map", J1, "; elliptic polar", J2)
print("area of the ellipse: ab * (area of unit disc) =", J1 * pi, "; by elliptic polar coordinates:", integrate(integrate(J2, r, 0, 1), t, 0, 2 * pi))
J3 = jacobian([r * sin(th) * cos(ph), r * sin(th) * sin(ph), r * cos(th)], [r, th, ph]).det().simplify_trig()
print("spherical Jacobian:", J3, "; volume of the unit ball:", integrate(integrate(integrate(J3, r, 0, 1), th, 0, pi), ph, 0, 2 * pi))
