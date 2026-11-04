# Exercise exr-lgt-sc-z2-mc: Wilson loops of the 3D Z2 gauge theory at beta = 0.2 by heat-bath Monte Carlo, against the strong-coupling series.
import numpy as np
rng = np.random.default_rng(int(41)); L, D = int(8), int(3)
par = np.indices((L,) * D).sum(axis=0) % 2
sh = lambda a, mu, k=1: np.roll(a, -k, axis=mu)
def sweep(s, beta):
    for mu in range(D):
        for p in (0, 1):
            st = sum(sh(s[nu], mu) * sh(s[mu], nu) * s[nu] + sh(sh(s[nu], mu), nu, -1) * sh(s[mu], nu, -1) * sh(s[nu], nu, -1) for nu in range(D) if nu != mu)
            new = np.where(rng.random(st.shape) < 1 / (1 + np.exp(-2 * beta * st)), 1, -1)
            s[mu] = np.where(par == p, new, s[mu])
def loop(s, R, T, mu=0, nu=1):               # product of links around R x T rectangles in the (mu, nu) plane
    w = np.ones(s.shape[1:])
    for i in range(R):
        w = w * sh(s[mu], mu, i) * sh(sh(s[mu], mu, i), nu, T)
    for j in range(T):
        w = w * sh(s[nu], nu, j) * sh(sh(s[nu], nu, j), mu, R)
    return w.mean()
beta = 0.2; s = rng.choice([-1, 1], size=(D,) + (L,) * D); meas = []
for k in range(1600):
    sweep(s, beta)
    if k >= 100:
        meas.append([loop(s, 1, 1), loop(s, 1, 2), loop(s, 2, 2)])
m = np.array(meas); err = m.reshape(10, -1, 3).mean(1).std(0, ddof=1) / np.sqrt(10); t = np.tanh(beta)
for (R, T), A, i in (((1, 1), 1, 0), ((1, 2), 2, 1), ((2, 2), 4, 2)):
    print('W(%d,%d): Monte Carlo %.5f +- %.5f, leading t^%d = %.5f' % (R, T, m[:, i].mean(), err[i], A, t**A))
print('plaquette with the cube correction t + 2 t^5 = %.5f' % (t + 2 * t**5))
