# Exercise exr-lgt-hb-kp: the Kennedy-Pendleton draw of a0 with density sqrt(1 - a0^2) exp(x a0), checked against <a0> = I_2(x)/I_1(x).
import numpy as np
rng = np.random.default_rng(int(91))
def kp(x, n):
    e = rng.random((4, n))
    Dd = -(np.log(e[0]) + np.log(e[1]) * np.cos(2 * np.pi * e[2])**2) / x
    ok = e[3]**2 <= 1 - Dd / 2
    return (1 - Dd)[ok], ok.mean()
for x in (1.0, 4.0, 10.0):
    a0, acc = kp(float(x), int(200000))
    print('x = beta k = %4.1f: <a0> = %.4f, I_2/I_1 = %.4f, acceptance %.3f' % (x, a0.mean(), bessel_I(2, x) / bessel_I(1, x), acc))
