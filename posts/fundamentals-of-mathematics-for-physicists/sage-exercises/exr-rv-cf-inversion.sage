# Exercise exr-rv-cf-inversion: Fourier inversion of the characteristic function of the sum of two dice,
# P(S = s) = (1/2 pi) int_0^{2 pi} e^{-its} phi_X(t)^2 dt, recovers the triangular table (6 - |s - 7|)/36.
t = var("t")
phi = sum(exp(I * k * t) for k in range(1, 7)) / 6
phiS = (phi ** 2).expand()
table = {}
for s in range(2, 13):
    table[s] = (integrate(exp(-I * t * s) * phiS, t, 0, 2 * pi) / (2 * pi)).simplify_full()
print("recovered probabilities:", table)
print("equal to (6 - |s - 7|)/36:", all(bool(table[s] == QQ(6 - abs(s - 7)) / 36) for s in range(2, 13)))
print("P(S = 0) from the same formula:", (integrate(phiS, t, 0, 2 * pi) / (2 * pi)).simplify_full())
