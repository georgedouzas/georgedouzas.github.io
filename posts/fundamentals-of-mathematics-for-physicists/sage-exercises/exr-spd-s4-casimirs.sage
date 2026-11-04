# Exercise exr-spd-s4-casimirs: spinors on S^4 = SO(5)/SO(4) from the SO(5) Casimir and dimension of Dolan (2003),
# eqs. (36)-(37), with (p, q) = (q + 1, q): C_2 + R/8 = (q + 2)^2 and d = 4 binom(q + 3, q).
C2 = lambda p, q: (p ^ 2 + q ^ 2) / 2 + 2 * p + q
d = lambda p, q: (p + q + 3) * (p - q + 1) * (p + 2) * (q + 1) / 6
for q in range(6):
    print("q = %d: C_2 + 3/2 = %s, (q+2)^2 = %d; d = %d, 4 binom(q+3, q) = %d" % (q, C2(q + 1, q) + 3/2, (q + 2) ^ 2, d(q + 1, q), 4 * binomial(q + 3, q)))
