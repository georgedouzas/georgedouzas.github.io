# Exercise exr-lgt-el-polyakov-torus: Polyakov loops of the 2D Z2 theory on the 4 x 2 torus (4 sites in space, 2 in time), enumerated.
import numpy as np
Lx, Lt = int(4), int(2); nl = 2 * Lx * Lt; n = Lx * Lt
bits = ((np.arange(2**nl)[:, None] >> np.arange(nl)) & 1) * -2 + 1
s = bits.reshape(-1, 2, Lx, Lt)                                   # s[:, 0] space links, s[:, 1] time links
P = (s[:, 0] * np.roll(s[:, 1], -1, 1) * np.roll(s[:, 0], -1, 2) * s[:, 1]).reshape(len(s), -1).sum(1)
pol = s[:, 1].prod(axis=2)                                        # Polyakov loop at each x
for beta in (0.5, 1.5):
    w = np.exp(beta * P); c, sh_ = np.cosh(beta), np.sinh(beta)
    print('beta = %.1f: <P(0)> = %.1e' % (beta, abs((w * pol[:, 0]).sum() / w.sum())))
    for R in (1, 2):
        m = R * Lt
        exact = (w * pol[:, 0] * pol[:, R]).sum() / w.sum()
        print('   <P(0) P(%d)> = %.8f, formula (s^m c^(n-m) + c^m s^(n-m))/(c^n + s^n) = %.8f' % (R, exact, (sh_**m * c**(n - m) + c**m * sh_**(n - m)) / (c**n + sh_**n)))
