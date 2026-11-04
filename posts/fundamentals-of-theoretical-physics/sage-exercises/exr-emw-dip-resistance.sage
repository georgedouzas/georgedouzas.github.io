# Exercise exr-emw-dip-resistance: radiation resistance of a short antenna (SI units).
Z0 = 376.73
for frac in [1/10, 1/50]:
    R = 2 * pi.n() / 3 * Z0 * frac^2
    print('l = lambda/%d: R_rad = %.3f ohm, current amplitude for 1 W: %.2f A' % (round(1 / frac), R, sqrt(2 / R)))
