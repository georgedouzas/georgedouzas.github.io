# Exercise exr-cq-dirichlet-test: int_{-pi}^{pi} D_N(s) f(s) ds with the Dirichlet kernel D_N = (1/2pi) sum_{|n|<=N} e^{ins}
# and f = 1/(2 - cos s) tends to f(0) = 1 geometrically; the error is 2 r^{N+1}/(sqrt(3)(sqrt(3) - 1)), r = 2 - sqrt(3).
s = var("s")
f = 1 / (2 - cos(s))
r = 2 - sqrt(3)
for N in [1, 2, 5, 10, 20]:
    D = (1 + 2 * sum(cos(n * s) for n in range(1, N + 1))) / (2 * pi)
    val = numerical_integral(D * f, -pi, pi)[0]
    print("N = %2d  integral = %.15f   1 - integral = %.3e   predicted error = %.3e" % (N, val, 1 - val, (2 * r ** (N + 1) / (sqrt(3) * (sqrt(3) - 1))).n()))
print("Fourier coefficient c_3 = (1/2pi) int f e^{-3is}:", numerical_integral(f * cos(3 * s), -pi, pi)[0] / (2 * pi).n(), " vs r^3/sqrt(3) =", (r ** 3 / sqrt(3)).n())
