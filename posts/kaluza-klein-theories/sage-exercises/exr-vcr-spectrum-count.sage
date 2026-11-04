# Exercise exr-vcr-spectrum-count: count the Kaluza-Klein modes of a torus T^2 with M <= Lambda, M^2 =
# |k2 - k1 tau|^2/(A tau_2), for the square (tau = i) and hexagonal (tau = e^{i pi/3}) tori of volume modulus A = 1,
# and compare with Weyl's law N(Lambda) ~ Vol Lambda^2/(4 pi) = pi A Lambda^2 (Vol = 4 pi^2 A for y in [0, 2 pi)^2).
def count(tau, Lam, A=1):
    t1, t2 = tau.real(), tau.imag()
    K = int(2 * Lam * sqrt(A / t2)) + 3
    return sum(1 for k1 in range(-K, K + 1) for k2 in range(-K, K + 1) if ((k2 - k1 * t1) ^ 2 + (k1 * t2) ^ 2) / (A * t2) <= Lam ^ 2)
sq = CC(0, 1); hx = CC(1 / 2, sqrt(3.0) / 2)
for Lam in [5, 10, 30]:
    print("Lambda = %2d: square %5d, hexagonal %5d, Weyl pi Lambda^2 = %.1f" % (Lam, count(sq, Lam), count(hx, Lam), float(pi * Lam ^ 2)))
lev = sorted(set(round(((k2 - k1 * 0.5) ^ 2 + (k1 * sqrt(3.0) / 2) ^ 2) / (sqrt(3.0) / 2), 8) for k1 in range(-3, 4) for k2 in range(-3, 4)))[:4]
print("hexagonal torus, lowest M^2 levels (A = 1):", lev, "; 2/sqrt(3) =", round(2 / sqrt(3.0), 8))
print("multiplicity of the first hexagonal level:", sum(1 for k1 in range(-3, 4) for k2 in range(-3, 4) if k1 ^ 2 - k1 * k2 + k2 ^ 2 == 1))
