# Exercise exr-dq-final-heavy: for hydrogen-like ions, the Dirac 1S binding energy, its ratio to the Schrodinger value, <r>/a_Z and the weight of
# the small component F^2/(G^2 + F^2) = f^2/(1 + f^2), f = (1 - gamma)/(Z alpha).
alpha = 1 / 137.035999166; me = 510.99895   # keV
print(" Z   binding (keV)  ratio   <r>/a_Z   small-component weight")
for Z in (1, 20, 40, 60, 80, 92):
    a = Z * alpha; gm = sqrt(1 - a^2); f = (1 - gm) / a
    print("%3d  %12.4f  %6.4f  %7.4f   %.3e" % (Z, me * (1 - gm), (1 - gm) / (a^2 / 2), (2 * gm + 1) / 2, f^2 / (1 + f^2)))
