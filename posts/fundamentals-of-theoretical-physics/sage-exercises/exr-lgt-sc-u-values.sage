# Exercise exr-lgt-sc-u-values: u(beta) at small beta for Z2, U(1), SU(2) and SU(3); for SU(3) by the Weyl integration formula.
import numpy as np
b = var('b')
for name, f in (('Z2', tanh(b)), ('U(1)', bessel_I(1, b) / bessel_I(0, b)), ('SU(2)', bessel_I(2, b) / bessel_I(1, b))):
    print(name, ': u =', taylor(f, b, 0, 3))
n = int(240); ph = 2 * np.pi * np.arange(n) / n
p1, p2 = np.meshgrid(ph, ph); p3 = -p1 - p2
z = [np.exp(1j * p) for p in (p1, p2, p3)]
vdm = abs(z[0] - z[1])**2 * abs(z[0] - z[2])**2 * abs(z[1] - z[2])**2      # Weyl density of SU(3) eigenphases
retr = (z[0] + z[1] + z[2]).real / 3
for bb in (0.01, 0.1, 1.0):
    w = vdm * np.exp(bb * retr)
    u = (w * retr).sum() / w.sum()
    print('SU(3), beta = %.2f: u = %.6f, u/beta = %.5f (1/18 = %.5f)' % (bb, u, u / bb, 1 / 18))
print('Haar check for SU(3): <|Tr U|^2> = %.6f, <(Tr U)^2> = %.1e' % ((vdm * abs(3 * retr + 1j * (z[0] + z[1] + z[2]).imag)**2).sum() / vdm.sum(), abs((vdm * (z[0] + z[1] + z[2])**2).sum() / vdm.sum())))
