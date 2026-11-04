# Exercise exr-dq-helium-ion: the Dirac 2P_{3/2} - 2P_{1/2} interval of He+ (Z = 2, reduced mass with M/m = 7294.30) against the
# complete theoretical value 175 593.50 MHz quoted by Eides et al.; the same comparison for hydrogen (10 969.04 MHz).
R = RealField(100)
alpha = R(1) / R("137.035999166"); me = R("510998.95069"); h = R("4.135667696e-15")
def f(n, j, Z):
    k = R(j) + R(1) / 2
    return (1 + (Z * alpha)^2 / (n - k + sqrt(k^2 - (Z * alpha)^2))^2)^(-R(1) / 2)
for name, Z, M, full in [("H", 1, R("1836.15267343"), R("10969.04")), ("He+", 2, R("7294.30"), R("175593.50"))]:
    mr = me / (1 + 1 / M)
    d = mr * (f(2, 3/2, Z) - f(2, 1/2, Z)) / h / 1e6
    print("%s: Dirac %.1f MHz, complete theory %.2f MHz, difference %.1f MHz = %.3f%%" % (name, d, full, full - d, 100 * (full - d) / full))
