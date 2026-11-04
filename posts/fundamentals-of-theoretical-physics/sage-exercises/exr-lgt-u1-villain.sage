# Exercise exr-lgt-u1-villain: matching the Villain coupling beta_V to the Wilson coupling beta through the plaquette.
import numpy as np
from scipy.integrate import quad
def villain_cos(bV, nmax=8):
    w = lambda t: sum(np.exp(-bV / 2 * (t - 2 * np.pi * k)**2) for k in range(-nmax, nmax + 1))
    return quad(lambda t: np.cos(t) * w(t), -np.pi, np.pi)[0] / quad(w, -np.pi, np.pi)[0]
for bV in (0.5, 1.5):
    print('check: Villain <cos theta> at beta_V = %.1f is %.8f, exp(-1/(2 beta_V)) = %.8f' % (bV, villain_cos(float(bV)), np.exp(-1 / (2 * float(bV)))))
for b in (0.5, 1, 2, 4, 10):
    bV = -1 / (2 * log(bessel_I(1, b) / bessel_I(0, b)))
    print('beta = %4.1f: beta_V = %.4f, beta - 1/2 = %.4f' % (b, bV, b - 0.5))
