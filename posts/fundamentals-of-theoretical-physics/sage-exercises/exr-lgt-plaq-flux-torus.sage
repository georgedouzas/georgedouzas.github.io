# Exercise exr-lgt-plaq-flux-torus: a uniform magnetic field on the periodic L x L lattice carries a total flux 2 pi k.
import numpy as np
L, k = int(4), int(1)
b = 2 * np.pi * k / L**2                     # flux per plaquette
th = np.zeros((2, L, L))
for x in range(L):
    for y in range(L):
        th[1, x, y] = b * x                  # theta_nu(x, y) = b x
        if x == L - 1:
            th[0, x, y] = -b * L * y         # the closing links in direction mu at x = L - 1
P = th[0] + np.roll(th[1], -1, axis=0) - np.roll(th[0], -1, axis=1) - th[1]
Pmod = (P + np.pi) % (2 * np.pi) - np.pi
print('plaquette angles reduced to (-pi, pi], in units of 2 pi / L^2:'); print(np.round(Pmod / b, 10))
print('total flux / 2 pi =', Pmod.sum() / (2 * np.pi))
print('sum of unreduced plaquette angles (always 0 on the torus):', round(P.sum(), 12))
