# Exercise exr-dq-uranium: 1S binding energy of hydrogen-like uranium (Z = 92) from the Dirac formula and from the Schrodinger equation.
alpha = 1 / 137.035999166; me = 510.99895   # keV
a = 92 * alpha
print("Z alpha = %.4f, (Z alpha)^2 = %.4f, sqrt(1 - (Z alpha)^2) = %.4f" % (a, a^2, sqrt(1 - a^2)))
print("Dirac binding %.2f keV, Schrodinger binding %.2f keV, ratio %.3f" % (me * (1 - sqrt(1 - a^2)), me * a^2 / 2, (1 - sqrt(1 - a^2)) / (a^2 / 2)))
