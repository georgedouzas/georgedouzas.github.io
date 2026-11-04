# Exercise exr-lgt-hb-acceptance: the acceptance of Creutz's SU(2) heat bath, pi I_1(x) / (2 sinh x), and its large-x form.
for x in (0.5, 1, 5, 10, 20, 50):
    acc = pi * bessel_I(1, x) / (2 * sinh(x))
    print('x = beta k = %4.1f: acceptance %.4f, sqrt(pi/(2x)) = %.4f' % (x, acc, sqrt(pi / (2 * x))))
