# Exercise exr-lgt-el-gauge-average: averaging a link over random gauge transformations drives it to zero; a plaquette trace is unchanged.
import numpy as np
rng = np.random.default_rng(int(51))
def rsu2(n):
    v = rng.normal(size=(n, 4)); return v / np.linalg.norm(v, axis=1)[:, None]
def qmul(a, b):
    c = np.empty(np.broadcast(a, b).shape)
    c[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    c[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return c
qinv = lambda a: a * np.array([1.0, -1, -1, -1])
U = rsu2(1)
for n in (100, 10000, 1000000):
    gx, gy = rsu2(n), rsu2(n)
    m = qmul(qmul(qinv(gx), U), gy).mean(0)
    print('n = %7d: |average of g(x)^-1 U g(y)| = %.5f, 1/sqrt(n) = %.5f' % (n, np.linalg.norm(m), 1 / np.sqrt(n)))
P = rsu2(1); g = rsu2(1000)
print('(1/2) Tr of g^-1 U_P g for 1000 random g: spread %.1e' % np.ptp(qmul(qmul(qinv(g), P), g)[:, 0]))
