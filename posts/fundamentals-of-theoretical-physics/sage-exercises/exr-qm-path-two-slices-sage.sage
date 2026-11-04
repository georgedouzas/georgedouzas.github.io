# Exercise exr-qm-path-two-slices-sage: the time-sliced Euclidean free path integral with N = 2, 3, 4 slices.
x, y = var("x y", domain="real"); hbar, m, eps = var("hbar m eps", domain="positive")
for N in [2, 3, 4]:
    zs = [y] + [var("z%d" % k) for k in range(1, N)] + [x]
    integrand = (m / (2 * pi * hbar * eps)) ** (N / 2) * exp(-sum(m * (zs[k + 1] - zs[k]) ** 2 / (2 * hbar * eps) for k in range(N)))
    for k in range(1, N):
        integrand = integrate(integrand, zs[k], -oo, oo)
    KE = sqrt(m / (2 * pi * hbar * N * eps)) * exp(-m * (x - y) ** 2 / (2 * hbar * N * eps))
    print("N =", N, ": sliced integral / K_E(x, y; N eps) =", (integrand / KE).canonicalize_radical())
