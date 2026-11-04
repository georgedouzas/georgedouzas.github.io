# Exercise exr-lgt-cl-2d-u1: continuum limit of the two-dimensional U(1) string tension, sigma = e^2 / 2.
for beta in (1, 3, 10, 30, 100, 300):
    sig = log(bessel_I(0, beta) / bessel_I(1, beta))
    print('beta = %5d: sigma a^2 = %.6f, beta sigma a^2 = %.6f' % (beta, sig, beta * sig))
print('lattice units at beta = 3: sigma a^2 = %.5f (Boudreau-Swanson beta = 6: 0.21076(7) by Monte Carlo)' % log(bessel_I(0, 3) / bessel_I(1, 3)))
