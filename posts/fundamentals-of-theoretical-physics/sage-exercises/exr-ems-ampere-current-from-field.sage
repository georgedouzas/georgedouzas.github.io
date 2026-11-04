# Exercise exr-ems-ampere-current-from-field: the current density of B = B0 exp(-s^2/a^2) (-y, x, 0) and its total.
x, y, z, B0, a, s = var('x y z B0 a s'); assume(a > 0, s > 0)
B = B0 * exp(-(x^2 + y^2) / a^2) * vector([-y, x, 0])
J = vector([diff(B[2], y) - diff(B[1], z), diff(B[0], z) - diff(B[2], x), diff(B[1], x) - diff(B[0], y)]).simplify_full()
print("J =", J)
Jz = 2 * B0 * (1 - s^2 / a^2) * exp(-s^2 / a^2)
print("J_z = 2 B0 (1 - s^2/a^2) exp(-s^2/a^2):", bool((J[2] - Jz.subs(s=sqrt(x^2 + y^2))).simplify_full() == 0))
print("total current:", integrate(2 * pi * s * Jz, s, 0, oo))
