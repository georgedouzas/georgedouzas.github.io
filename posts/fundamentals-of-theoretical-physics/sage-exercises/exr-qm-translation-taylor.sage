# Exercise exr-qm-translation-taylor: exp(-a d/dx) shifts polynomials; exp(-i a p/hbar) in momentum space shifts a Gaussian.
x, a, p = var("x a p", domain="real"); hbar, s = var("hbar s", domain="positive")
f = 3 * x ** 5 - x ** 3 + 2 * x - 7
print("sum_n (-a)^n f^(n)/n! - f(x - a) =", (sum((-a) ** n / factorial(n) * diff(f, x, n) for n in range(6)) - f.subs(x=x - a)).expand())
psi = (2 * pi * s ** 2) ** (-1 / 4) * exp(-x ** 2 / (4 * s ** 2))
pt = (2 * s ** 2 / (pi * hbar ** 2)) ** (1 / 4) * exp(-s ** 2 * p ** 2 / hbar ** 2)
back = integrate(exp(I * p * x / hbar) * exp(-I * a * p / hbar) * pt, p, -oo, oo) / sqrt(2 * pi * hbar)
print("inverse transform of e^(-i a p/hbar) psi~ divided by psi(x - a):", (back / psi.subs(x=x - a)).canonicalize_radical())
