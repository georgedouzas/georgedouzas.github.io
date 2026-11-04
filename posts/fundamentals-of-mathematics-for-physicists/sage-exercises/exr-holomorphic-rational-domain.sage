# Exercise exr-holomorphic-rational-domain: the domain of holomorphy of (z^2 + 1)/(z^4 - 1) after cancellation, and
# the zeros of cos z (the poles of tan z).
z = var("z")
f = (z^2 + 1)/(z^4 - 1)
print("simplified:", f.simplify_rational())
print("poles after cancellation:", [r for r, m in (z^2 - 1).roots(ring=QQbar)])
print("cos z = 0 solutions:", solve(cos(z) == 0, z, to_poly_solve=True))
x, y = var("x y", domain="real")
c = cos(x + I*y).expand()
print("|cos(x + i y)|^2 = cos^2 x + sinh^2 y:", bool((c.real_part()^2 + c.imag_part()^2 - (cos(x)^2 + sinh(y)^2)).simplify_full().trig_simplify() == 0), " so zeros need sinh y = 0 and cos x = 0")
