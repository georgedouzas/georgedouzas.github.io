# Exercise exr-ems-capacitance-earth: the Earth and the ionosphere as a spherical capacitor (SI units).
eps0 = 8.854e-12; R = 6.371e6; h = 6e4; V = 3e5
C = 4 * pi.n() * eps0 * R * (R + h) / h
Cp = eps0 * 4 * pi.n() * R^2 / h
print("C = %.4f F, plate approximation %.4f F, isolated Earth %.3e F" % (C, Cp, 4 * pi.n() * eps0 * R))
print("charge at 300 kV: %.2e C" % (C * V))
