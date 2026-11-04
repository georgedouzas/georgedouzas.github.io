# Exercise exr-ub-unbounded-norm: domains of the derivative on the circle in the Fourier picture: the partial sums
# sum n^2 |<e_n, f>|^2 and sum n^4 |<e_n, f>|^2 for f = x (a jump when periodised) and g = pi^2 - x^2 (a kink).
x = var("x", domain="real")
c = lambda f, n: (integrate(f * exp(-I * n * x), x, -pi, pi) / sqrt(2 * pi)).simplify_full()
f, g = x, pi ** 2 - x ** 2
cf = lambda n: 2 * pi / n ** 2            # |<e_n, f>|^2, n != 0
cg = lambda n: 8 * pi / n ** 4            # |<e_n, g>|^2, n != 0
print("|<e_n, x>|^2 for n = 1, 2, 3:", [(abs(c(f, n)) ** 2).simplify_full() for n in [1, 2, 3]], "; |<e_n, pi^2 - x^2>|^2:", [(abs(c(g, n)) ** 2).simplify_full() for n in [1, 2, 3]])
for N in [10, 100, 1000]:
    s1 = sum(2 * n ** 2 * cf(n) for n in range(1, N + 1)); s2 = sum(2 * n ** 2 * cg(n) for n in range(1, N + 1)); s3 = sum(2 * n ** 4 * cg(n) for n in range(1, N + 1))
    print("N = %4d: sum n^2 |<e_n, f>|^2 = %10.3f   sum n^2 |<e_n, g>|^2 = %10.4f   sum n^4 |<e_n, g>|^2 = %10.1f" % (N, s1.n(), s2.n(), s3.n()))
n = var("n", domain="integer")
print("limits: sum n^2 |<e_n, g>|^2 =", 2 * sum(n ** 2 * cg(n), n, 1, oo), "; the other two series diverge (their terms, 4 pi and 16 pi for each |n|, do not tend to 0)")
