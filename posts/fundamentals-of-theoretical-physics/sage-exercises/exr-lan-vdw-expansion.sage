# Exercise exr-lan-vdw-expansion: the reduced van der Waals equation p_r = 8 T_r/(3 V_r - 1) - 3/V_r^2 near the critical
# point, V_r = 1 + phi, T_r = 1 + t: p_r = 1 + 4t - 6 t phi - (3/2) phi^3 + ..., and the coexistence curve
# phi_l,g = -+ 2 sqrt(-t) from the Maxwell equal-area construction on the cubic.
phi, t = var("phi t")
pr = 8 * (1 + t) / (3 * (1 + phi) - 1) - 3 / (1 + phi) ** 2
ser = pr.taylor(phi, 0, 3).expand()
keep = sum(c * phi ** k * t ** j for (c, k, j) in [(ser.coefficient(phi, k).coefficient(t, j), k, j) for k in range(4) for j in range(2)] if (k, j) in [(0, 0), (0, 1), (1, 1), (3, 0)])
print("p_r to the relevant order:", keep)
print("coefficient of phi^2:", ser.coefficient(phi, 2), "(it vanishes at t = 0, and 9 t phi^2 ~ t^2 is subleading on the coexistence curve phi ~ sqrt(-t))")
# cubic isotherm p = 1 + 4t - 6 t phi - (3/2) phi^3 is odd in phi about phi = 0 at p = 1 + 4t, so equal areas give
# coexistence at that pressure with phi = +-sqrt(-4t)
cub = -6 * t * phi - 3 / 2 * phi ** 3
print("roots of -6 t phi - (3/2) phi^3 = 0:", solve(cub == 0, phi), "; equal-area integral int_{-a}^{a} =", integrate(cub, phi, -2 * sqrt(-t), 2 * sqrt(-t)))
