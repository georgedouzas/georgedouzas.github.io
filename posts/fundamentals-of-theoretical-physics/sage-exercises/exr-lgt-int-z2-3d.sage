# Exercise exr-lgt-int-z2-3d: area and perimeter laws of the 3D Z2 gauge theory on either side of beta_c, by heat-bath Monte Carlo.
import numpy as np
rng = np.random.default_rng(int(102)); L, D = int(8), int(3)
par = np.indices((L,) * D).sum(axis=0) % 2
sh = lambda a, mu, k=1: np.roll(a, -k, axis=mu)
def sweep(s, beta):
    for mu in range(D):
        for p in (0, 1):
            st = sum(sh(s[nu], mu) * sh(s[mu], nu) * s[nu] + sh(sh(s[nu], mu), nu, -1) * sh(s[mu], nu, -1) * sh(s[nu], nu, -1) for nu in range(D) if nu != mu)
            s[mu] = np.where(par == p, np.where(rng.random(st.shape) < 1 / (1 + np.exp(-2 * beta * st)), 1, -1), s[mu])
def loop(s, R, T):
    w = np.ones(s.shape[1:])
    for mu, nu in ((0, 1), (1, 2), (2, 0)):
        x = np.ones(s.shape[1:])
        for i in range(R):
            x = x * sh(s[mu], mu, i) * sh(sh(s[mu], mu, i), nu, T)
        for j in range(T):
            x = x * sh(s[nu], nu, j) * sh(sh(s[nu], nu, j), mu, R)
        w = w + x
    return (w - 1).mean() / 3
for beta in (0.25, 1.0):
    s = np.ones((D,) + (L,) * D); acc = {}
    for k in range(1300):
        sweep(s, beta)
        if k >= 100:
            for RT in ((1, 1), (1, 2), (2, 2)):
                acc.setdefault(RT, []).append(loop(s, *RT))
    W = {RT: np.mean(v) for RT, v in acc.items()}
    print('beta = %.2f:' % beta, '  '.join('W%d%d = %.5f' % (R, T, W[(R, T)]) for (R, T) in W))
    chi = -np.log(W[(2, 2)] * W[(1, 1)] / W[(1, 2)]**2)
    print('   Creutz ratio chi(2,2) = %.4f; strong coupling -log tanh(beta) = %.4f' % (chi, -np.log(np.tanh(beta))))
    print('   -log W / perimeter: %s; dilute flipped links give 2 exp(-8 beta) = %.5f' % (' '.join('%.4f' % (-np.log(W[RT]) / (2 * sum(RT))) for RT in W), 2 * np.exp(-8 * beta)))
    print('   -log W / area:      %s' % ' '.join('%.4f' % (-np.log(W[RT]) / (RT[0] * RT[1])) for RT in W))
