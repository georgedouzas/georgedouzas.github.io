# Exercise exr-lgt-wl-lattice-units: a string tension of 0.18 GeV^2 in lattice units, and the spacing it implies.
hbarc = 0.1973269804                         # GeV fm
sigma = 0.18                                 # GeV^2
print('sigma = %.4f GeV/fm, sqrt(sigma) = %.0f MeV' % (sigma / hbarc, 1000 * sqrt(sigma)))
for a in (0.2, 0.1, 0.05):
    print('a = %.2f fm: sigma a^2 = %.4f' % (a, sigma * (a / hbarc)^2))
for sig_hat in (0.21, 0.07, 0.035):
    print('sigma a^2 = %.3f: a = %.4f fm' % (sig_hat, hbarc * sqrt(sig_hat / sigma)))
