# Exercise exr-emx-dl-ionosphere: plasma frequencies and wavelengths (SI).
e = 1.602176634e-19; me = 9.1093837e-31; eps0 = 8.8541878128e-12; c = 299792458
for N in [1e10, 1e12, 1e29]:
    fp = sqrt(N * e^2 / (eps0 * me)) / (2 * pi)
    print('n =', N, ': f_p =', n(fp, digits=3), 'Hz, lambda_p =', n(c / fp, digits=3), 'm')
