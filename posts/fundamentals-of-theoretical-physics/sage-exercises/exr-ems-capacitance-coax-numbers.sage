# Exercise exr-ems-capacitance-coax-numbers: capacitance per metre of a coaxial cable (SI units).
eps0 = 8.854e-12; a = 0.5e-3; b = 1.75e-3; V = 50
C = 2 * pi.n() * eps0 / log(b / a)
print("C = %.2f pF/m, lambda = C V = %.2f nC/m" % (C * 1e12, C * V * 1e9))
