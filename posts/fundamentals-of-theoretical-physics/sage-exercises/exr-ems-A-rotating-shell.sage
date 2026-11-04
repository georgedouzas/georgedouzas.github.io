# Exercise exr-ems-A-rotating-shell (Macchi et al., Problem 5.12): rotating charged shell, omega along e_z.
x, y, z, sigma, a, w = var('x y z sigma a w'); assume(a > 0)
X = vector([x, y, z]); r = sqrt(x^2 + y^2 + z^2); W = vector([0, 0, w])
def curl(F): return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
def div(F): return sum(diff(F[i], v) for i, v in enumerate([x, y, z]))
Ain = sigma * a / 3 * W.cross_product(X); Aout = sigma * a^4 / 3 * W.cross_product(X) / r^3
print("div A_in =", div(Ain), "; div A_out =", div(Aout).simplify_full())
print("B_in =", curl(Ain))
m = 4 * pi / 3 * sigma * w * a^4 * vector([0, 0, 1])
Bdip = (3 * m.dot_product(X) * X / r^2 - m) / (4 * pi * r^3)
print("B_out is the dipole field with m = 4 pi sigma w a^4/3:", bool((curl(Aout) - Bdip).simplify_full() == 0))
print("continuity of A at r = a (point (a,0,0)):", (Ain - Aout).subs(x=a, y=0, z=0).simplify_full())
jump = (curl(Aout) - curl(Ain)).subs(x=a, y=0, z=0).simplify_full()
print("n x (B_out - B_in) at (a, 0, 0):", vector([1, 0, 0]).cross_product(jump), "; K = sigma w x X there:", (sigma * W.cross_product(X)).subs(x=a, y=0, z=0))
