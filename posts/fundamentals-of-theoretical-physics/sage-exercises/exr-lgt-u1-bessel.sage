# Exercise exr-lgt-u1-bessel: the one-plaquette average <cos theta> = I_1(beta)/I_0(beta) of compact U(1).
b, th = var('b theta')
u = bessel_I(1, b) / bessel_I(0, b)
print('series:', taylor(u, b, 0, 5))
for bb in (2, 8, 20):
    num = numerical_integral(cos(th) * exp(bb * cos(th)), 0, 2*pi)[0] / numerical_integral(exp(bb * cos(th)), 0, 2*pi)[0]
    print('beta = %.1f: integral %.8f, I_1/I_0 %.8f, 1 - 1/(2 beta) %.5f' % (bb, num, u(b=bb).n(), 1 - 1 / (2 * bb)))
