# Exercise exr-lgt-su2-random: random SU(2) elements by rejection from the 4-cube to the unit ball, then normalised.
import numpy as np
rng = np.random.default_rng(int(21))
x = rng.uniform(-1, 1, size=(int(200000), int(4)))
r2 = (x**2).sum(1); keep = x[r2 <= 1]
a = keep / np.linalg.norm(keep, axis=1)[:, None]
print('acceptance %.4f, expected pi^2/32 = %.4f' % (len(keep) / len(x), np.pi**2 / 32))
print('<a0> = %.4f, <a0^2> = %.4f (1/4), <a0^4> = %.4f (1/8)' % (a[:, 0].mean(), (a[:, 0]**2).mean(), (a[:, 0]**4).mean()))
h, edges = np.histogram(a[:, 0], bins=10, range=(float(-1), float(1)), density=True); mid = (edges[1:] + edges[:-1]) / 2
print('histogram of a0 against (2/pi) sqrt(1 - a0^2):')
for m, v in zip(mid, h):
    print('  a0 = %+.1f: %.3f  %.3f' % (m, v, 2 / np.pi * np.sqrt(1 - m**2)))
