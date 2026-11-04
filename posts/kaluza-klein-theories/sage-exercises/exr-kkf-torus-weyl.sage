# Exercise exr-kkf-torus-weyl: number of four-dimensional Dirac fermions with mass below E from a seven-dimensional
# Dirac fermion on the cube torus T^3 (R_i = 1, N = 2 fermions per lattice point), periodic and antiperiodic in all
# directions, compared with the Weyl-law estimate N (4 pi/3) E^3 R_1 R_2 R_3.
def count(E, a):
    M = ceil(E) + 1
    return 2 * len([1 for m1 in range(-M, M + 1) for m2 in range(-M, M + 1) for m3 in range(-M, M + 1)
                    if (m1 + a) ** 2 + (m2 + a) ** 2 + (m3 + a) ** 2 < E ** 2])
for E in (2, 5, 10, 20):
    nP, nA, w = count(E, 0), count(E, QQ(1) / 2), 2 * 4 * pi.n() / 3 * E ** 3
    print("E R = %2d: periodic %6d, antiperiodic %6d, Weyl law %9.1f, ratios %.3f %.3f" % (E, nP, nA, w, nP / w, nA / w))
