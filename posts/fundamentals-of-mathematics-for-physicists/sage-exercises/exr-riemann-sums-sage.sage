# Exercise exr-riemann-sums-sage: lower and upper Riemann sums of x^2 on [0, 1] in exact arithmetic for n = 4, 10,
# 100, 1000, their closed forms, and their common limit 1/3 = the integral.
x, n = var("x n")
for m in [4, 10, 100, 1000]:
    h = QQ(1) / m
    L = sum((k * h) ** 2 * h for k in range(m)); U = sum(((k + 1) * h) ** 2 * h for k in range(m))
    print("n =", m, ": lower", L, "=", L.n(digits=6), ", upper", U, "=", U.n(digits=6), ", gap", U - L)
print("closed forms: lower = (n-1)n(2n-1)/(6n^3), upper = n(n+1)(2n+1)/(6n^3); both -> 1/3:", limit((n - 1) * n * (2 * n - 1) / (6 * n ** 3), n=oo), limit(n * (n + 1) * (2 * n + 1) / (6 * n ** 3), n=oo))
print("integral:", integrate(x ** 2, x, 0, 1))
