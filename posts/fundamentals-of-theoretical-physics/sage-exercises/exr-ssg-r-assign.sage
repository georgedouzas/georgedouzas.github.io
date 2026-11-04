# Exercise exr-ssg-r-assign: a U(1)_R symmetry exists when every monomial of the superpotential has total R-charge 2.
# Each monomial prod Phi_i^{n_i} gives a linear equation sum n_i R_i = 2; we solve the system for four superpotentials.
def r_charges(name, n, monomials):
    A = matrix(QQ, [list(m) for m in monomials]); b = vector(QQ, [2] * len(monomials))
    try:
        x0 = A.solve_right(b)
    except ValueError:
        print("%-38s no R-charge assignment" % name); return None
    K = A.right_kernel()
    print("%-38s R = %s + span%s" % (name, tuple(x0), [tuple(v) for v in K.basis()]))
    return x0, K.dimension()
# Nath (8.316): W = c1 Phi1 + c2 Phi2^2 + c3 Phi3^3
a = r_charges("c1 Phi1 + c2 Phi2^2 + c3 Phi3^3", 3, [(1, 0, 0), (0, 2, 0), (0, 0, 3)])
print("[PASS] Nath's example: R(Phi1, Phi2, Phi3) = (2, 1, 2/3), uniquely:", a is not None and tuple(a[0]) == (2, 1, QQ(2) / 3) and a[1] == 0)
# O'Raifeartaigh: W = -k Phi1 + m Phi2 Phi3 + y/2 Phi1 Phi3^2
b = r_charges("O'Raifeartaigh", 3, [(1, 0, 0), (0, 1, 1), (1, 0, 2)])
print("[PASS] O'Raifeartaigh: R(Phi1, Phi2, Phi3) = (2, 2, 0), uniquely:", b is not None and tuple(b[0]) == (2, 2, 0) and b[1] == 0)
# Wess-Zumino: W = m/2 Phi^2 + y/6 Phi^3
c = r_charges("Wess-Zumino m Phi^2 + y Phi^3", 1, [(2,), (3,)])
print("[PASS] the massive interacting Wess-Zumino model has no R-symmetry:", c is None)
# a Dirac mass: W = m Phi+ Phi-
d = r_charges("m Phi+ Phi-", 2, [(1, 1)])
print("[PASS] W = m Phi+ Phi- leaves a one-parameter family (R-symmetry times the ordinary U(1) with charges (1, -1)):",
      d is not None and d[1] == 1)
