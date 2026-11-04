# Exercise exr-spd-nk-euler: Euler characteristics |W_G|/|W_H| of the six-dimensional nearly-Kaehler cosets.
W = lambda typ: WeylGroup(typ).cardinality()
cases = [("G2/SU(3)", W(["G", 2]), W(["A", 2])), ("Sp(4)/(SU(2) x U(1))", W(["C", 2]), W(["A", 1])), ("SU(3)/(U(1) x U(1))", W(["A", 2]), 1)]
for name, wg, wh in cases:
    print("%-22s |W_G| = %2d, |W_H| = %d, chi = %d, generations chi/2 = %d" % (name, wg, wh, wg / wh, wg / wh / 2))
