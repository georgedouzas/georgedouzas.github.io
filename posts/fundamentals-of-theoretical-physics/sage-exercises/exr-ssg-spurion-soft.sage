# Exercise exr-ssg-spurion-soft: a spurion S = 1 + xi1 theta theta + xi1* thetadagger thetadagger + xi2 theta theta thetadagger thetadagger
# multiplies the kinetic and superpotential terms of one chiral superfield. The F-dependent part of the Lagrangian is
#   |F|^2 + xi1 phi F* + xi1* phi* F + (W'(phi) F + c.c.) + xi2 |phi|^2 + (xi1 W(phi) + c.c.),
# and eliminating F gives the scalar potential V = -L. We treat phi, phi*, F, F*, xi1, xi1* as independent symbols.
R = PolynomialRing(QQ, "phi, phib, F, Fb, x1, x1b, x2, m, mb, y, yb"); phi, phib, F, Fb, x1, x1b, x2, m, mb, y, yb = R.gens()
W = m / 2 * phi^2 + y / 6 * phi^3; Wb = mb / 2 * phib^2 + yb / 6 * phib^3
Wp = W.derivative(phi); Wbp = Wb.derivative(phib)
L = F * Fb + x1 * phi * Fb + x1b * phib * F + Wp * F + Wbp * Fb + x2 * phi * phib + x1 * W + x1b * Wb
# the auxiliary equations dL/dF = 0 = dL/dFb
sol = {Fb: -(x1b * phib + Wp), F: -(x1 * phi + Wbp)}
print("[PASS] the auxiliary field equations are solved by F* = -(W' + xi1* phi*), F = -(W'* + xi1 phi):",
      L.derivative(F).subs(sol) == 0 and L.derivative(Fb).subs(sol) == 0)
V = -L.subs(sol)
susy = Wp * Wbp
soft = V - susy
print("soft part of V =", soft)
expected = (x1 * x1b - x2) * phi * phib + x1 * (phi * Wp - W) + x1b * (phib * Wbp - Wb)
print("[PASS] V = |W'|^2 + (|xi1|^2 - xi2)|phi|^2 + [xi1 (phi W' - W) + c.c.]:", soft == expected)
print("[PASS] B and A terms: xi1 (phi W' - W) = xi1 (m/2 phi^2 + 2 * y/6 phi^3), so A0 = 2 B0 = 2 xi1:",
      x1 * (phi * Wp - W) == x1 * (1 * m / 2 * phi^2 + 2 * y / 6 * phi^3))
