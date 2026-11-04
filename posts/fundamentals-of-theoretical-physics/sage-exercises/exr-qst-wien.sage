# Exercise exr-qst-wien: maxima of the Planck spectrum per unit frequency (x = 3(1 - e^{-x})) and per unit wavelength
# (x = 5(1 - e^{-x})), with x = hbar omega/(k_B T) or h c/(lambda k_B T), and the Wien constants.
x = var("x")
x3 = find_root(x - 3 * (1 - exp(-x)), 1, 5); x5 = find_root(x - 5 * (1 - exp(-x)), 1, 6)
print("frequency peak: x = %.6f ; wavelength peak: x = %.6f" % (x3, x5))
h, c, kB = 6.62607015e-34, 299792458, 1.380649e-23
b = h * c / (kB * x5)
print("Wien constant b = h c/(k_B x) = %.6e m K" % b)
for name, T in [("Sun", 5772), ("CMB", 2.725), ("human body", 310)]:
    print("%s, T = %s K: lambda_max = %.4g m, nu_max = %.4g Hz" % (name, T, b / T, x3 * kB * T / h))
