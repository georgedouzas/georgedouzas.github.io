# Exercise exr-emw-larmor-thomson: the Thomson cross-section and the photon mean free path in the solar core (SI units).
re = 2.818e-15
sT = 8 * pi.n() / 3 * re^2
print('sigma_T = %.3e m^2, mean free path at n = 1e30 m^-3: %.3e m' % (sT, 1 / (1e30 * sT)))
