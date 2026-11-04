# Exercise exr-lgt-int-su2: four-dimensional SU(2): plaquette by heat-bath Monte Carlo on 4^4, strong and weak coupling, and the scale.
import numpy as np
rng = np.random.default_rng(int(103)); L, D = int(4), int(4)
par = np.indices((L,) * D).sum(axis=0) % 2
def qmul(a, b):
    c = np.empty(np.broadcast(a, b).shape)
    c[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    c[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return c
qinv = lambda a: a * np.array([1.0, -1, -1, -1])
sh = lambda a, mu, k=1: np.roll(a, -k, axis=mu)
def staple(U, mu):
    return sum(qmul(qmul(sh(U[nu], mu), qinv(sh(U[mu], nu))), qinv(U[nu]))
               + qmul(qmul(qinv(sh(sh(U[nu], mu), nu, -1)), qinv(sh(U[mu], nu, -1))), sh(U[nu], nu, -1)) for nu in range(D) if nu != mu)
def heatbath(st, beta):                         # Kennedy-Pendleton for W = U V, then U = W V^-1
    k = np.linalg.norm(st, axis=-1); V = st / k[..., None]; x = beta * k
    a0 = np.empty(k.shape); todo = np.ones(k.shape, bool)
    while todo.any():
        e = rng.random((4, todo.sum())); Dd = -(np.log(e[0]) + np.log(e[1]) * np.cos(2 * np.pi * e[2])**2) / x[todo]
        ok = e[3]**2 <= 1 - Dd / 2; idx = np.flatnonzero(todo)[ok]; a0.flat[idx] = 1 - Dd[ok]; todo.flat[idx] = False
    v = rng.normal(size=k.shape + (3,)); v /= np.linalg.norm(v, axis=-1)[..., None]
    W = np.concatenate([a0[..., None], np.sqrt(1 - a0**2)[..., None] * v], axis=-1)
    return qmul(W, qinv(V))
plaq = lambda U: np.mean([qmul(qmul(qmul(U[m], sh(U[n], m)), qinv(sh(U[m], n))), qinv(U[n]))[..., 0].mean() for m in range(D) for n in range(m + 1, D)])
for beta in (0.5, 4.0):
    U = np.zeros((D,) + (L,) * D + (4,)); U[..., 0] = 1; h = []
    for k in range(250):
        for mu in range(D):
            for p in (0, 1):
                U[mu] = np.where((par == p)[..., None], heatbath(staple(U, mu), beta), U[mu])
        if k >= 50: h.append(plaq(U))
    print('beta = %.1f: <(1/2) Tr U_P> = %.4f; strong coupling beta/4 = %.4f; weak coupling 1 - 3/(4 beta) = %.4f' % (beta, np.mean(h), beta / 4, 1 - 3 / (4 * beta)))
hbarc = 0.1973269804
sig_hat = float(exp(12 * pi^2 / 11 - 6 * pi^2 * 2.5 / 11))
print('beta = 2.5: one-loop scaling sigma a^2 = %.4f, a = %.3f fm with sigma = 0.18 GeV^2' % (sig_hat, hbarc * np.sqrt(sig_hat / 0.18)))
