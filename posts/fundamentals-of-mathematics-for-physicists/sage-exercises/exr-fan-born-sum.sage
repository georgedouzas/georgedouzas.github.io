# Exercise exr-fan-born-sum: K f(x) = x int_0^1 y f(y) dy; K^n = 3^(1-n) K, the Born series for f = 1, and the exact solution.
x, y, lam, c = var('x y lam c')
K = lambda f: x * integrate(y * f.subs(x=y), y, 0, 1)
f = SR(1); Kf = K(f)
print("K 1 =", Kf, ", K^2 1 =", K(Kf), ", K^3 1 =", K(K(Kf)))
series = 1 + sum(lam^n * 3^(1 - n) * Kf for n in range(1, 40))
exact = 1 + lam * x / 2 / (1 - lam / 3)
print("partial sum (39 terms) at lam = 1, x = 1: %.12f; closed form: %.12f" % (series.subs(lam=1, x=1).n(), exact.subs(lam=1, x=1).n()))
u = 1 + c * x
sol = solve((u - lam * K(u) - 1).coefficient(x) == 0, c)
print("solving u - lam K u = 1 with u = 1 + c x:", sol)
