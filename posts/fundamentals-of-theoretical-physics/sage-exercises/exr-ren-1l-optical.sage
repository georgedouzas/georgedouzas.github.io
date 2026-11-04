# Exercise exr-ren-1l-optical: the imaginary part of the renormalised bubble above threshold and the optical theorem
# for two identical scalars, m = 1, s = 9, lambda = 1.
m, s, lam = 1, 9, 1
x = var('x')
r1, r2 = sorted([r.rhs() for r in solve(m^2 - x * (1 - x) * s == 0, x)])
print("roots:", r1, r2, "  length of the interval:", (r2 - r1).simplify_full(), "= sqrt(1 - 4m^2/s):", bool(r2 - r1 == sqrt(1 - 4 * m^2 / s)))
ImB = pi * (r2 - r1) / (16 * pi^2)
ImM = lam^2 / 2 * ImB
sigma = lam^2 / (32 * pi * s)
print("Im M(s, 0) = %.8f   s sqrt(1 - 4m^2/s) sigma = %.8f" % (ImM.n(), (s * sqrt(1 - 4 * m^2 / s) * sigma).n()))
