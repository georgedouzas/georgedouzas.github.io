# Exercise exr-lgt-wa-one-plaquette: z(beta) = integral of exp((beta/N) Re Tr U) over the group, for Z2, U(1), SU(2).
b, th, psi = var('b theta psi')
zZ2 = (exp(b) + exp(-b)) / 2
zU1 = lambda bb: numerical_integral(exp(bb * cos(th)), 0, 2*pi)[0] / (2*pi)
print('Z2:   z =', zZ2.simplify_full(), '= cosh(beta)')
for bb in (0.5, 1, 3): print('U(1), beta = %.1f: z = %.10f, I_0(beta) = %.10f' % (bb, zU1(bb), bessel_I(0, bb)))
for bb in (0.5, 1, 3):                       # SU(2): Weyl measure (2/pi) sin^2 psi d psi, (1/2) Tr U = cos psi
    zS = numerical_integral(2/pi * sin(psi)^2 * exp(bb * cos(psi)), 0, pi)[0]
    print('SU(2), beta = %.1f: z = %.10f, 2 I_1(beta)/beta = %.10f' % (bb, zS, 2 * bessel_I(1, bb) / bb))
