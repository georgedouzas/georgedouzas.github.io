# Exercise exr-lgt-el-z2-bound: a link of the 2D Z2 gauge theory with a source J on every link; exact enumeration on the 3 x 3 torus.
import numpy as np
L = int(3)
bits = ((np.arange(2**18)[:, None] >> np.arange(18)) & 1) * -2 + 1
s = bits.reshape(-1, 2, L, L)
P = (s[:, 0] * np.roll(s[:, 1], -1, 1) * np.roll(s[:, 0], -1, 2) * s[:, 1]).reshape(len(s), -1).sum(1)
links = s.reshape(len(s), -1).sum(1)
sx = s[:, 0, 0, 0] + s[:, 1, 0, 0] + s[:, 0, L - 1, 0] + s[:, 1, 0, L - 1]     # the four links at the site (0, 0)
beta = 1.0
for J in (0.0, 0.02, 0.1, 0.3):
    w = np.exp(beta * P + J * links)
    m = (w * s[:, 0, 0, 0]).sum() / w.sum(); m2 = (w * s[:, 0, 0, 0] * np.tanh(J * sx)).sum() / w.sum()
    print('J = %.2f: <sigma_l> = %.6f, <sigma_l tanh(J s_x)> = %.6f, bound tanh(4J) = %.6f' % (J, m, m2, np.tanh(4 * J)))
