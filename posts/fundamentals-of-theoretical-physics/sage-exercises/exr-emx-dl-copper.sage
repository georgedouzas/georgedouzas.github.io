# Exercise exr-emx-dl-copper: Drude parameters of copper (SI).
e = 1.602176634e-19; me = 9.1093837e-31; eps0 = 8.8541878128e-12; hbar = 1.054571817e-34; c = 299792458
N = 8.5e28; s0 = 5.96e7
wp = sqrt(N * e^2 / (eps0 * me))
g = eps0 * wp^2 / s0
print('omega_p =', n(wp, digits=3), '1/s; hbar omega_p =', n(hbar * wp / e, digits=3), 'eV; gamma =', n(g, digits=3), '1/s')
w = 2 * pi * c / 600e-9
eps = 1 - wp^2 / (w * (w + I * g))
print('omega =', n(w, digits=3), '1/s; eps =', CDF(eps))
