# Exercise exr-lgt-sc-su2-crossing: where the strong-coupling SU(2) string tension meets the one-loop scaling curve.
b = var('b')
u = bessel_I(2, b) / bessel_I(1, b)
sc0 = -log(u); sc1 = -log(u) - 4 * u^4
rg = exp(12 * pi^2 / 11 - 6 * pi^2 * b / 11)
x0 = find_root(sc0 - rg, 1.5, 3); x1 = find_root(sc1 - rg, 1.5, 3)
print('crossing with -log u: beta = %.4f, sigma a^2 = %.4f' % (x0, rg(b=x0)))
print('crossing with -log u - 4u^4: beta = %.4f, sigma a^2 = %.4f' % (x1, rg(b=x1)))
print('at beta = 2.5: -log u - 4u^4 = %.4f, scaling = %.4f' % (sc1(b=2.5), rg(b=2.5)))
