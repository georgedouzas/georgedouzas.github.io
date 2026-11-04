# Exercise exr-hso-circle-zeta: the circle with every tool of the chapter: orthonormality and Parseval for x (zeta(2)),
# Laplacian eigenfunctions and the Poisson heat trace, the spectral zeta function 2 zeta(2s) with its determinant and
# regularised sums, and the periodic Casimir energy -pi/(6L).
x = var("x", domain="real")
e = lambda n: exp(I * n * x) / sqrt(2 * pi)
print("Gram matrix of e_{-1}, e_0, e_1:", matrix(SR, 3, 3, lambda j, k: integrate(conjugate(e(j - 1)) * e(k - 1), x, -pi, pi)).list())
n = var("n", domain="integer")
print("Parseval for x: ||x||^2 / (4 pi) =", integrate(x ** 2, x, -pi, pi) / (4 * pi), "= zeta(2) =", zeta(2))
print("-e_n'' = n^2 e_n for n = -3..3:", all(bool(-diff(e(k), x, 2) == k ** 2 * e(k)) for k in range(-3, 4)))
RF = RealField(80); t = RF(1) / 10
print("heat trace at t = 0.1: %.15f  Poisson: %.15f" % (sum(exp(-t * k ** 2) for k in range(-60, 61)), sqrt(RF(pi) / t) * sum(exp(-RF(pi) ** 2 * k ** 2 / t) for k in range(-5, 6))))
s = var("s")
zD = 2 * zeta(2 * s)
print("det_zeta(-Delta') = exp(-zeta_Delta'(0)) =", exp(-diff(zD, s).subs(s=0)).n(digits=15), "; 4 pi^2 =", (4 * pi ** 2).n(digits=15))
print("regularised sum n^2 = 2 zeta(-2) =", 2 * zeta(-2), "; sum |n| = 2 zeta(-1) =", 2 * zeta(-1))
L = var("L", domain="positive")
print("Casimir energy E = (pi/L) 2 zeta(-1) =", pi / L * 2 * zeta(-1))
