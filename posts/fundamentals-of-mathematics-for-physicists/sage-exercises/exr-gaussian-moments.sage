# Exercise exr-gaussian-moments: the Gaussian integral and its relatives: int e^{-a x^2} dx = sqrt(pi/a), the moments
# int x^{2n} e^{-x^2} dx = Gamma(n + 1/2) = (2n-1)!! sqrt(pi)/2^n, odd moments vanish, and the completed square
# int e^{-a x^2 + b x} dx = sqrt(pi/a) e^{b^2/(4a)}.
x, a, b = var("x a b")
assume(a > 0)
print("int e^{-a x^2} dx =", integrate(exp(-a * x ** 2), x, -oo, oo))
print("even moments int x^{2n} e^{-x^2} dx for n = 0..3:", [integrate(x ** (2 * k) * exp(-x ** 2), x, -oo, oo) for k in range(4)])
print("double factorial formula (2n-1)!! sqrt(pi)/2^n for n = 0..3:", [prod(range(1, 2 * k, 2)) * sqrt(pi) / 2 ** k for k in range(4)])
print("odd moments vanish:", [integrate(x ** (2 * k + 1) * exp(-x ** 2), x, -oo, oo) for k in range(3)])
print("int e^{-a x^2 + b x} dx =", integrate(exp(-a * x ** 2 + b * x), x, -oo, oo).simplify_full(), "= sqrt(pi/a) e^{b^2/(4a)}")
