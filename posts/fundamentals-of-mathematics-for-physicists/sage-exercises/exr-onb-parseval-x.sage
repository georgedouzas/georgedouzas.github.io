# Exercise exr-onb-parseval-x: Fourier coefficients <e_n, f> of f(x) = x and g(x) = x^2 on (-pi, pi) in the orthonormal
# basis e_n = e^{inx}/sqrt(2 pi), Parseval's identity as the series for zeta(2) and zeta(4), and the partial sums.
x = var("x", domain="real")
c = lambda f, n: (integrate(f * exp(-I * n * x), x, -pi, pi) / sqrt(2 * pi)).simplify_full()
f, g = x, x ** 2
print("<e_n, x> for n = 1, 2, 3:", [c(f, n) for n in [1, 2, 3]], "; <e_0, x^2> =", c(g, 0), "; <e_n, x^2> for n = 1, 2, 3:", [c(g, n) for n in [1, 2, 3]])
n = var("n", domain="integer")
print("||x||^2 =", integrate(f ** 2, x, -pi, pi), " = 4 pi sum 1/n^2 gives sum 1/n^2 =", (integrate(f ** 2, x, -pi, pi) / (4 * pi)).simplify_full(), "=", sum(1 / n ** 2, n, 1, oo))
print("||x^2||^2 =", integrate(g ** 2, x, -pi, pi), " = 2 pi^5/9 + 16 pi sum 1/n^4 gives sum 1/n^4 =", ((integrate(g ** 2, x, -pi, pi) - 2 * pi ** 5 / 9) / (16 * pi)).simplify_full(), "=", sum(1 / n ** 4, n, 1, oo))
for h, name in [(f, "x"), (g, "x^2")]:
    full = integrate(h ** 2, x, -pi, pi)
    part = sum(abs(c(h, k)) ** 2 for k in range(-10, 11))
    print("partial Parseval sum |n| <= 10 for", name, ":", part.n(digits=8), " norm^2:", full.n(digits=8), " ratio:", (part / full).n(digits=6))
