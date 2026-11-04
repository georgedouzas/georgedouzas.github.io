# Exercise exr-lgt-2d-u1-torus: compact U(1) on the L x L torus, Z = sum_n I_n(beta)^(L^2), and the finite-size plaquette.
def plaquette(beta, L, nmax=10):
    I = lambda k: bessel_I(abs(k), beta)
    num = sum(I(k)^(L*L - 1) * (I(k - 1) + I(k + 1)) / 2 for k in range(-nmax, nmax + 1))
    return num / sum(I(k)^(L*L) for k in range(-nmax, nmax + 1))
for beta in (1, 3):
    print('beta = %d: infinite lattice I_1/I_0 = %.8f' % (beta, bessel_I(1, beta) / bessel_I(0, beta)))
    for L in (1, 2, 3, 4, 8):
        print('    L = %d: <cos theta_P> = %.8f' % (L, plaquette(beta, L)))
