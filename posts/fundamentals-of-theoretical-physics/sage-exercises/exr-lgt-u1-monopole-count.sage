# Exercise exr-lgt-u1-monopole-count: monopoles of a random compact U(1) configuration on the periodic 4^3 lattice.
import numpy as np
rng = np.random.default_rng(int(12)); L = int(4)
th = rng.uniform(-np.pi, np.pi, size=(3, L, L, L))
sh = lambda a, mu: np.roll(a, -1, axis=mu)
def plaq(m, n, red=True):
    p = th[m] + sh(th[n], m) - sh(th[m], n) - th[n]
    return (p + np.pi) % (2 * np.pi) - np.pi if red else p
def flux(red):                               # oriented sum over the six faces of the cube at x
    return (sh(plaq(1, 2, red), 0) - plaq(1, 2, red)) - (sh(plaq(0, 2, red), 1) - plaq(0, 2, red)) + (sh(plaq(0, 1, red), 2) - plaq(0, 1, red))
n = flux(True) / (2 * np.pi)
print('unreduced angles: largest |sum over a cube| = %.1e' % np.abs(flux(False)).max())
print('monopole numbers are integers:', np.allclose(n, np.round(n)))
vals, counts = np.unique(np.round(n).astype(int), return_counts=True)
print('monopole numbers and how many cubes:', dict(zip(vals.tolist(), counts.tolist())))
print('net monopole number on the torus:', int(np.round(n).sum()))
