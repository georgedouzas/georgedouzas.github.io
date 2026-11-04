# Exercise exr-lgt-int-2d-u1: two-dimensional compact U(1) by heat-bath Monte Carlo (von Mises draws) on a 12 x 12 torus at beta = 2.
import numpy as np
rng = np.random.default_rng(int(101)); L, beta = int(12), 2.0
par = np.indices((L, L)).sum(axis=0) % 2
sh = lambda a, mu, k=1: np.roll(a, -k, axis=mu)
th = np.zeros((2, L, L))
def staple(th, mu):
    nu = 1 - mu
    return np.exp(1j * (sh(th[nu], mu) - sh(th[mu], nu) - th[nu])) + np.exp(1j * (-sh(sh(th[nu], mu), nu, -1) - sh(th[mu], nu, -1) + sh(th[nu], nu, -1)))
W = {(1, 1): [], (2, 2): [], (2, 3): []}
for k in range(4100):
    for mu in (0, 1):
        for p in (0, 1):
            st = staple(th, mu)
            new = rng.vonmises(0.0, beta * np.abs(st)) - np.angle(st)
            th[mu] = np.where(par == p, new, th[mu])
    if k >= 100:
        P = th[0] + sh(th[1], 0) - sh(th[0], 1) - th[1]
        for (R, T) in W:
            W[(R, T)].append(np.cos(sum(sh(sh(P, 0, i), 1, j) for i in range(R) for j in range(T))).mean())
u = bessel_I(1, beta) / bessel_I(0, beta)
for (R, T), v in W.items():
    v = np.array(v); err = v.reshape(20, -1).mean(1).std(ddof=1) / np.sqrt(20)
    print('W(%d,%d) = %.4f +- %.4f; u^%d = %.4f' % (R, T, v.mean(), err, R * T, u**(R * T)))
print('string tension in lattice units -log u = %.4f; in units of e^2 (beta = 1/(e^2 a^2)): beta * (-log u) = %.4f -> 1/2' % (-log(u), -beta * log(u)))
