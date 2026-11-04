# Exercise exr-fan-lp-borderline: zeta values, int_0^1 x^(-p/4) dx, Gaussian L^p norms, and the moments of log(1/x).
x, p, t = var('x p t'); assume(p > 0, p < 4)
print("zeta(2), zeta(3), zeta(4) =", zeta(2), zeta(3).n(digits=8), zeta(4))
print("int_0^1 x^(-p/4) dx =", integrate(x^(-p / 4), x, 0, 1))
for q in (1, 2, 3):
    print("|exp(-x^2/2)|_%d =" % q, (integrate(exp(-q * x^2 / 2), x, -oo, oo))^(1 / q))
forget(); assume(t > 0)
for q in (1, 2, 5, 1/2):
    print("int_0^1 log(1/x)^%s dx = %s (Gamma(%s) = %s)" % (q, numerical_integral(lambda u: log(1 / u)^q, 0, 1)[0], q + 1, gamma(q + 1).n()))
