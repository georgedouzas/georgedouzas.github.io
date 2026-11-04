# Exercise exr-vcp-dirac-energy: the energy of N flux quanta on the sphere of radius R. The uniform field
# F = N/(2 q R^2) vol has F_{mn} F^{mn} = 2 (N/(2qR^2))^2, and the Maxwell energy (1/4) int F_{mn} F^{mn} vol = pi N^2/(2 q^2 R^2):
# quadratic in N and decreasing with R, so flux pushes the sphere to expand.
th, ph, N, q, R = var("theta phi N q R", domain="positive")
Fthph = N / (2 * q * R**2) * R**2 * sin(th)          # component F_{theta phi}
ginv = diagonal_matrix([1 / R**2, 1 / (R**2 * sin(th)**2)])
F2 = 2 * Fthph**2 * ginv[0, 0] * ginv[1, 1]         # F_{mn} F^{mn}
print("F_{mn} F^{mn} =", F2.simplify_full())
E = integrate(integrate(F2 / 4 * R**2 * sin(th), th, 0, pi), ph, 0, 2 * pi)
print("(1/4) int F_{mn} F^{mn} vol =", E.simplify_full())
print("flux int F =", integrate(integrate(Fthph, th, 0, pi), ph, 0, 2 * pi))
