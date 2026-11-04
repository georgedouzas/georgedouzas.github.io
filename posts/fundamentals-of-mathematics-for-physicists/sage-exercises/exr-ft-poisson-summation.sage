# Exercise exr-ft-poisson-summation: Poisson summation for the Gaussian, sum_n e^{-2 pi^2 n^2/sigma^2} =
# sigma/sqrt(2 pi) sum_n e^{-sigma^2 n^2/2}, checked numerically for several sigma.
for s in [1, 2, 3, 5]:
    lhs = sum(exp(-2 * pi ** 2 * n ** 2 / s ** 2) for n in range(-30, 31)).n(digits=15)
    rhs = (s / sqrt(2 * pi) * sum(exp(-s ** 2 * n ** 2 / 2) for n in range(-30, 31))).n(digits=15)
    print("sigma =", s, ": sum f(2 pi n) =", lhs, "; (1/2pi) sum fhat(n) =", rhs, "; agree to 1e-12:", bool(abs(lhs - rhs) < 1e-12))
