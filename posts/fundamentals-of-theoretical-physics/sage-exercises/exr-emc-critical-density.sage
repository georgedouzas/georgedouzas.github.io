# Exercise exr-emc-critical-density: critical electron density n_c = eps0 m omega^2/e^2 (SI).
e = 1.602176634e-19; me = 9.1093837e-31; eps0 = 8.8541878128e-12; c = 299792458
for lam in [1e-6, 0.35e-6]:
    w = 2 * pi * c / lam
    nc = eps0 * me * w^2 / e^2
    print(lam * 1e6, 'um: n_c =', n(nc, digits=3), 'm^-3; aluminium/n_c =', n(1.8e29 / nc, digits=3))
