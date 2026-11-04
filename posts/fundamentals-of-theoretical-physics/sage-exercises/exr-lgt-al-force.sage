# Exercise exr-lgt-al-force: the string tension as a force.
sigma_GeV_fm = 0.18 / 0.1973269804
GeV = 1.602176634e-10; fm = 1e-15; g = 9.80665
F = sigma_GeV_fm * GeV / fm
print('sigma = %.3f GeV/fm = %.3e N = %.1f tonnes-force' % (sigma_GeV_fm, F, F / g / 1000))
print('energy stored in 1 fm of string: %.0f MeV; length at which it reaches 2 m_pi = 0.28 GeV: %.2f fm' % (1000 * sigma_GeV_fm, 0.28 / sigma_GeV_fm))
