# Exercise exr-ren-ce-yang: the logarithmic slope of Yang's magnetisation m(K) = (1 - sinh(2K)^-4)^(1/8) at K - K_c = 1e-2, 1e-4, 1e-6.
R = RealField(120)
Kc = asinh(R(1)) / 2
m = lambda K: (1 - sinh(2 * K)^(-4))^(R(1) / 8)
for dk in [R(1e-2), R(1e-4), R(1e-6)]:
    h = dk * R(1e-3)
    slope = (log(m(Kc + dk + h)) - log(m(Kc + dk - h))) / (log(dk + h) - log(dk - h))
    print("K - K_c = %.0e: slope %.6f" % (dk, slope))
print("prefactor (8 sqrt 2)^(1/8) = %.4f, m / (K - K_c)^(1/8) at 1e-6: %.4f" % ((8 * sqrt(R(2)))^(R(1) / 8), m(Kc + R(1e-6)) / R(1e-6)^(R(1) / 8)))
