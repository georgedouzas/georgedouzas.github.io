# Exercise exr-lgt-su2-one-plaq: the one-plaquette average <(1/2) Tr U> = I_2(beta)/I_1(beta) for SU(2).
psi = var('psi')
for b in (1, 5, 20, 100):
    num = numerical_integral(cos(psi) * sin(psi)^2 * exp(b * cos(psi)), 0, pi)[0]
    den = numerical_integral(sin(psi)^2 * exp(b * cos(psi)), 0, pi)[0]
    print('beta = %4.1f: integral %.8f, I_2/I_1 %.8f, 1 - 3/(2 beta) %.5f' % (b, num / den, bessel_I(2, b) / bessel_I(1, b), 1 - 3 / (2 * b)))
