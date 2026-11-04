# Exercise exr-ems-potential-segment (Macchi et al., Problem 2.14): potential of a charged segment |z| <= c; ellipsoidal equipotentials.
lam, c, s, z, t = var('lam c s z t'); assume(c > 0, s > 0)
r1 = sqrt(s^2 + (z + c)^2); r2 = sqrt(s^2 + (z - c)^2)
phi_formula = lam / (4 * pi) * log((r1 + r2 + 2 * c) / (r1 + r2 - 2 * c))
for (sv, zv) in [(1, 0), (0.5, 2), (2, -1), (0.3, 0.7)]:
    print("s = %s, z = %s: integral %.10f, formula %.10f" % (sv, zv, numerical_integral(1 / (4 * pi * sqrt(sv^2 + (zv - t)^2)), -1, 1)[0], phi_formula.subs(lam=1, c=1, s=sv, z=zv).n()))
# on the ellipse r1 + r2 = 2A (A = 2): the potential is constant
A = 2; B = sqrt(A^2 - 1); u = var('u')
vals = [phi_formula.subs(lam=1, c=1, s=B * sin(uu), z=A * cos(uu)).n() for uu in [0.3, 1.0, 1.7, 2.5]]
print("potential at four points of the ellipse z = 2 cos u, s = sqrt(3) sin u:", vals)
