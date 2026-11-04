# Exercise exr-fan-h1-norm-circle: the H^1 norm of f = sin x + cos(2x)/2 on the circle, directly and from the coefficients.
x = var('x'); k = var('k')
f = sin(x) + cos(2 * x) / 2
direct = integrate(f^2, x, 0, 2 * pi) + integrate(diff(f, x)^2, x, 0, 2 * pi)
c = lambda kk: integrate(f * exp(-I * kk * x), x, 0, 2 * pi) / (2 * pi)
coeff = 2 * pi * sum((1 + kk^2) * abs(c(kk))^2 for kk in range(-4, 5))
print("|f|_2^2 + |f'|_2^2 =", direct, "; 2 pi sum (1 + k^2)|c_k|^2 =", coeff.simplify_full(), "; equal:", bool(direct == coeff))
print("nonzero coefficients:", [(kk, c(kk)) for kk in range(-4, 5) if c(kk) != 0])
