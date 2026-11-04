# Exercise exr-emw-lw-pulse: the field pulse of a 1 GeV electron passing at 1 cm (SI units).
e, eps0, c, me = 1.602e-19, 8.854e-12, 2.998e8, 0.511e6
g = 1e9 / me; b = 0.01
E = g * e / (4 * pi.n() * eps0 * b^2)
print('gamma = %.0f, peak transverse field %.3e V/m, duration b/(gamma c) = %.2e s' % (g, E, b / (g * c)))
