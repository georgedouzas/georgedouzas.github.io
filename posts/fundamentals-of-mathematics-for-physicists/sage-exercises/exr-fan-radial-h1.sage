# Exercise exr-fan-radial-h1: for f = |x|^(-a) on the unit ball of R^3, int |grad f|^2 and int f^2 in polar coordinates.
r, a = var('r a'); assume(r > 0)
assume(a < 1 / 2, a > 0)
print("int |grad f|^2 = 4 pi int a^2 r^(-2a-2) r^2 dr =", (4 * pi * integrate(a^2 * r^(-2 * a - 2) * r^2, r, 0, 1)).simplify_full())
forget(); assume(r > 0); assume(a < 3 / 2, a > 0)
print("int f^2 = 4 pi int r^(-2a) r^2 dr =", (4 * pi * integrate(r^(-2 * a) * r^2, r, 0, 1)).simplify_full())
for av in (0.25, 0.49, 0.5, 1.0, 1.49):
    gi = numerical_integral(lambda rr: av^2 * rr^(-2 * av), 1e-12, 1)[0]
    print("a = %.2f: int_eps^1 a^2 r^(-2a) dr with eps = 1e-12: %.4g" % (av, gi))
