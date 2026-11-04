# Exercise exr-lgt-hb-u1-sampler: the accept-reject heat bath of compact U(1) (uniform theta, accept with exp(x (cos theta - 1))).
import numpy as np
rng = np.random.default_rng(int(92))
for x in (0.5, 2.0, 10.0):
    th = rng.uniform(-np.pi, np.pi, int(400000)); keep = th[rng.random(th.size) < np.exp(float(x) * (np.cos(th) - 1))]
    print('x = %4.1f: <cos theta> = %.4f, I_1/I_0 = %.4f; acceptance %.4f, exp(-x) I_0(x) = %.4f' % (x, np.cos(keep).mean(), bessel_I(1, x) / bessel_I(0, x), keep.size / th.size, exp(-x) * bessel_I(0, x)))
print('numpy von Mises draw at x = 10: <cos> = %.4f' % np.cos(rng.vonmises(0.0, 10.0, int(400000))).mean())
