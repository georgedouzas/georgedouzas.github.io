# Exercise exr-gaussian-polar: the Gaussian integral by squaring and changing to polar coordinates: I^2 = int int
# e^{-x^2-y^2} dx dy = int_0^{2pi} int_0^oo e^{-r^2} r dr dphi = pi, with the Jacobian r of the polar map.
x, y, r, phi = var("x y r phi")
J = jacobian([r * cos(phi), r * sin(phi)], [r, phi])
print("Jacobian of the polar map:", J.rows(), "; determinant:", J.det().simplify_trig())
inner = integrate(exp(-r ** 2) * r, r, 0, oo)
print("int_0^oo e^{-r^2} r dr =", inner, "; times 2 pi:", 2 * pi * inner)
I = integrate(exp(-x ** 2), x, -oo, oo)
print("I = int e^{-x^2} dx =", I, "; I^2 =", (I ** 2).simplify_full(), "= pi:", bool(I ** 2 == pi))
print("the Cartesian double integral directly:", integrate(integrate(exp(-x ** 2 - y ** 2), x, -oo, oo), y, -oo, oo))
