# Exercise exr-ems-ampere-wire-check: divergence and curl of the field of a thick wire, inside and outside.
x, y, z, I, a = var('x y z I a'); assume(a > 0)
s2 = x^2 + y^2
Bin = I / (2 * pi * a^2) * vector([-y, x, 0]); Bout = I / (2 * pi * s2) * vector([-y, x, 0])
def div(F): return sum(diff(F[i], v) for i, v in enumerate([x, y, z])).simplify_full()
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)]).simplify_full()
print("inside: div =", div(Bin), ", curl =", curl(Bin), "= J = I/(pi a^2) e_z")
print("outside: div =", div(Bout), ", curl =", curl(Bout))
