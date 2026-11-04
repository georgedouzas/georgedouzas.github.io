# Exercise exr-lgt-z2-random-gauge: a random Z2 gauge transformation on a 4^3 lattice changes many links and no plaquette.
import numpy as np
rng = np.random.default_rng(int(8)); L, D = int(4), int(3)
s = rng.choice([-1, 1], size=(D, L, L, L)); eps = rng.choice([-1, 1], size=(L, L, L))
sh = lambda a, mu: np.roll(a, -1, axis=mu)
t = np.stack([eps * s[mu] * sh(eps, mu) for mu in range(D)])
def plaqs(s):
    return np.stack([s[m] * sh(s[n], m) * sh(s[m], n) * s[n] for m in range(D) for n in range(m + 1, D)])
print('links changed: %d of %d' % ((t != s).sum(), s.size))
print('plaquettes changed: %d of %d' % ((plaqs(t) != plaqs(s)).sum(), plaqs(s).size))
print('average plaquette before and after: %.4f %.4f' % (plaqs(s).mean(), plaqs(t).mean()))
