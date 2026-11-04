# Exercise exr-lgt-2d-continuum-su2: the continuum string tension of two-dimensional SU(2), sigma = 3 g^2 / 8.
for beta in (2, 10, 50, 200, 500):
    sig = -log(bessel_I(2, beta) / bessel_I(1, beta))
    print('beta = %4d: sigma a^2 = %.6f, beta sigma a^2 = %.6f' % (beta, sig, beta * sig))
print('beta = 4/(g^2 a^2), so sigma = (beta sigma a^2) g^2 / 4 -> (3/2)(g^2/4) = 3 g^2/8 = g^2 C_F / 2 with C_F = 3/4')
