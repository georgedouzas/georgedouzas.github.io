# Exercise exr-lgt-wl-stokes: the abelian lattice Stokes theorem on a random U(1) configuration.
import numpy as np
rng = np.random.default_rng(int(31)); L = int(6)
th = rng.uniform(-np.pi, np.pi, size=(2, L, L))
P = th[0] + np.roll(th[1], -1, 0) - np.roll(th[0], -1, 1) - th[1]        # plaquette angles theta_P(x, y)
def loop_angle(x, y, R, T):                   # counterclockwise around the R x T rectangle with corner (x, y)
    s = sum(th[0, (x + i) % L, y] for i in range(R)) + sum(th[1, (x + R) % L, (y + j) % L] for j in range(T))
    s -= sum(th[0, (x + i) % L, (y + T) % L] for i in range(R)) + sum(th[1, x, (y + j) % L] for j in range(T))
    return s
for (x, y, R, T) in ((0, 0, 1, 1), (1, 2, 3, 2), (2, 0, 4, 5)):
    enclosed = sum(P[(x + i) % L, (y + j) % L] for i in range(R) for j in range(T))
    print('loop %dx%d at (%d,%d): sum of link angles %.10f, sum of enclosed plaquette angles %.10f' % (R, T, x, y, loop_angle(x, y, R, T), enclosed))
