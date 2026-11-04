# Exercise exr-stc-kernel-eigen: the eigenvalue equation of the Green's operator, int_0^1 (min(x, y) - xy) sin(n pi y) dy =
# sin(n pi x)/(n pi)^2, and the partial sums of the kernel expansion at (1/3, 1/2).
x, y = var("x y", domain="real")
assume(x > 0, x < 1)
for n in [1, 2, 3]:
    Gs = integrate((y - x * y) * sin(n * pi * y), y, 0, x) + integrate((x - x * y) * sin(n * pi * y), y, x, 1)
    print("n =", n, ": G sin(n pi x) =", Gs.simplify_full(), "; equals sin(n pi x)/(n pi)^2:", bool((Gs - sin(n * pi * x) / (n * pi) ** 2).simplify_full() == 0))
x0, y0 = QQ(1) / 3, QQ(1) / 2
exact = min(x0, y0) - x0 * y0
for N in [1, 3, 10]:
    S = sum(2 * sin(k * pi * x0) * sin(k * pi * y0) / (k * pi) ** 2 for k in range(1, N + 1))
    print("N =", N, ": partial sum =", S.n(digits=6), " exact min(x, y) - xy =", exact, "=", exact.n(digits=6))
