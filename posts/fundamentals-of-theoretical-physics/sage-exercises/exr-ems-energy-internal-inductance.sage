# Exercise exr-ems-energy-internal-inductance: magnetic energy inside a uniform wire.
I, a, s = var('I a s', domain='positive')
W = integrate((I * s / (2 * pi * a^2))^2 / 2 * 2 * pi * s, s, 0, a)
print("energy per unit length:", W, "; inductance per unit length 2W/I^2 =", (2 * W / I^2))
print("SI: mu0/(8 pi) = %.1f nH/m" % (1.2566e-6 / (8 * pi.n()) * 1e9))
