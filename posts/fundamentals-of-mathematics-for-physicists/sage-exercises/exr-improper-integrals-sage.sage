# Exercise exr-improper-integrals-sage: convergence of improper integrals: int_1^oo dx/x^p converges exactly for
# p > 1, int_0^1 dx/x^p for p < 1; int_0^oo e^{-x} x^n dx = n!; int_{-oo}^{oo} dx/(1 + x^2) = pi.
x = var("x")
for p in [QQ(1) / 2, 1, 2, 3]:
    try:
        v = integrate(1 / x ** p, x, 1, oo)
    except ValueError:
        v = "divergent"
    print("int_1^oo x^(-%s) dx =" % p, v)
for p in [QQ(1) / 2, 1, 2]:
    try:
        v = integrate(1 / x ** p, x, 0, 1)
    except ValueError:
        v = "divergent"
    print("int_0^1 x^(-%s) dx =" % p, v)
print("int_0^oo e^{-x} x^n dx for n = 0..5:", [integrate(exp(-x) * x ** k, x, 0, oo) for k in range(6)], "= n!")
print("int_{-oo}^{oo} dx/(1 + x^2) =", integrate(1 / (1 + x ** 2), x, -oo, oo))
