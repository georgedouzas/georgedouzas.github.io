# Exercise exr-lgt-hb-z2-3d: the statistical error of the plaquette from blocks of increasing length, away from and near beta_c.
import numpy as np
rng = np.random.default_rng(int(93)); L, D = int(8), int(3)
par = np.indices((L,) * D).sum(axis=0) % 2
sh = lambda a, mu, k=1: np.roll(a, -k, axis=mu)
def sweep(s, beta):
    for mu in range(D):
        for p in (0, 1):
            st = sum(sh(s[nu], mu) * sh(s[mu], nu) * s[nu] + sh(sh(s[nu], mu), nu, -1) * sh(s[mu], nu, -1) * sh(s[nu], nu, -1) for nu in range(D) if nu != mu)
            s[mu] = np.where(par == p, np.where(rng.random(st.shape) < 1 / (1 + np.exp(-2 * beta * st)), 1, -1), s[mu])
plaq = lambda s: np.mean([(s[m] * sh(s[n], m) * sh(s[m], n) * s[n]).mean() for m in range(D) for n in range(m + 1, D)])
for beta in (0.4, 0.76):
    s = np.ones((D,) + (L,) * D); h = []
    for k in range(2100):
        sweep(s, beta)
        if k >= 100: h.append(plaq(s))
    h = np.array(h)
    errs = [h.reshape(-1, b).mean(1).std(ddof=1) / np.sqrt(len(h) // b) for b in (1, 5, 25, 100)]
    tau = 0.5 * (errs[-1] / errs[0])**2
    print('beta = %.2f: <sigma_P> = %.4f; error with blocks of 1, 5, 25, 100 sweeps: %s; tau_int ~ %.1f sweeps' % (beta, h.mean(), ' '.join('%.5f' % e for e in errs), tau))
