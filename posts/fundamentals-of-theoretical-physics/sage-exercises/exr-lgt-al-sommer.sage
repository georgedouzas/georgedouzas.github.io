# Exercise exr-lgt-al-sommer: the Sommer scale r0 from the Cornell potential, r0^2 F(r0) = 1.65.
hbarc = 0.1973269804
alpha = pi / 12                               # the universal 1/R term of a vibrating string
for sqrt_sigma in (0.42, 0.44):
    s = sqrt_sigma^2 / hbarc^2                 # fm^-2
    r0 = sqrt((1.65 - alpha) / s)
    print('sqrt(sigma) = %.0f MeV: r0 = %.4f fm, r0 sqrt(sigma) = %.4f' % (1000 * sqrt_sigma, r0, r0 * sqrt(s)))
