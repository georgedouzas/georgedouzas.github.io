# Exercise exr-lgt-2d-z2-torus: the Z2 gauge theory on the 3 x 3 torus by enumerating all 2^18 configurations.
import numpy as np
L = int(3); n = L * L
bits = ((np.arange(2**18)[:, None] >> np.arange(18)) & 1) * -2 + 1
s = bits.reshape(-1, 2, L, L)
P = s[:, 0] * np.roll(s[:, 1], -1, 1) * np.roll(s[:, 0], -1, 2) * s[:, 1]
Psum = P.reshape(len(s), -1).sum(1)
for beta in (0.5, 1.0, 2.0):
    w = np.exp(beta * Psum)
    mc = (w * Psum).sum() / w.sum() / n
    c, sh_ = np.cosh(beta), np.sinh(beta)
    formula = (sh_ * c**(n - 1) + c * sh_**(n - 1)) / (c**n + sh_**n)
    print('beta = %.1f: enumeration %.8f, formula %.8f, infinite lattice tanh %.8f' % (beta, mc, formula, np.tanh(beta)))
print('Z / 2^18 at beta = 1: %.6f; cosh(1)^9 + sinh(1)^9 = %.6f' % (np.exp(1.0 * Psum).mean(), np.cosh(1)**9 + np.sinh(1)**9))
