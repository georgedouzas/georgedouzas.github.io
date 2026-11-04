# Exercise exr-lgt-link-random-gauge: a random gauge transformation of a random SU(2) configuration leaves every Tr U_P unchanged.
import numpy as np
rng = np.random.default_rng(int(3)); L = int(4)
def rsu2(*shape):                            # random SU(2) as quaternions (a0, a1, a2, a3), uniform on S^3
    v = rng.normal(size=shape + (4,)); return v / np.linalg.norm(v, axis=-1)[..., None]
def qmul(a, b):
    c = np.empty(np.broadcast(a, b).shape)
    c[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    c[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return c
qinv = lambda a: a * np.array([1.0, -1, -1, -1])
sh = lambda a, mu: np.roll(a, -1, axis=mu)   # a(x + mu-hat), periodic
def traces(U):                               # (1/2) Tr U_{01}(x) = a0 of the plaquette
    return qmul(qmul(qmul(U[0], sh(U[1], 0)), qinv(sh(U[0], 1))), qinv(U[1]))[..., 0]
U = rsu2(2, L, L); g = rsu2(L, L)
V = np.stack([qmul(qmul(qinv(g), U[mu]), sh(g, mu)) for mu in (0, 1)])   # U_mu(x) -> g(x)^{-1} U_mu(x) g(x + mu)
print('largest change of a plaquette trace:', np.abs(traces(U) - traces(V)).max())
W = U.copy()                                 # the same transformation site by site, as in a program
for x in range(L):
    for y in range(L):
        for mu in (0, 1):
            W[mu, x, y] = qmul(qinv(g[x, y]), W[mu, x, y])
            xm, ym = ((x - 1) % L, y) if mu == 0 else (x, (y - 1) % L)
            W[mu, xm, ym] = qmul(W[mu, xm, ym], g[x, y])
print('site-by-site update equals the formula:', np.abs(W - V).max() < 1e-12)
print('a link itself changes (gauge variant): %.3f' % np.abs(U[0, 0, 0, 0] - V[0, 0, 0, 0]))
