# Exercise exr-cff-int-families: Euler numbers and family counts of the internal spaces of the chapter; cosets by the Weyl
# group formula chi = |W_S|/|W_R|, Calabi-Yau spaces from Hodge numbers, the orbifold from h11 = 9 + 27.
W = lambda t: WeylGroup(t).order()
rows = [("G_2/SU(3)", W(["G", 2]) / W(["A", 2]), "CSDR matching"),
        ("Sp(4)/(SU(2) x U(1))", W(["C", 2]) / W(["A", 1]), "CSDR matching"),
        ("SU(3)/(U(1) x U(1))", W(["A", 2]) / 1, "CSDR matching"),
        ("quintic", 2 * (1 - 101), "twisted index"),
        ("Y/G", 2 * (1 - 4), "twisted index"),
        ("T^6/Z_3", 2 * ((9 + 27) - 0), "projection + fixed points")]
for name, chi, mech in rows:
    print("%-22s chi = %5d   families = %3d   (%s)" % (name, chi, abs(chi) / 2, mech))
