# Exercise exr-lgt-z2-excitations: counting plaquettes per link and the weak-coupling expansion 1 - 8 exp(-4(d - 1) beta).
import numpy as np
from itertools import combinations
for d in (2, 3, 4):
    L = 4; nsites = L**d
    nlinks = d * nsites; nplaq = d * (d - 1) // 2 * nsites
    per_link = 4 * nplaq // nlinks             # each plaquette has 4 links
    print('d = %d: links per site %d, plaquettes per site %d, plaquettes through a link %d, flipping one link costs %d beta' % (d, d, d * (d - 1) // 2, per_link, 2 * per_link))
    if d > 2:
        print('       <sigma_P> = 1 - 2 * %d * (%d/%d) * exp(-%d beta) = 1 - 8 exp(-%d beta)' % (per_link, nlinks, nplaq, 2 * per_link, 2 * per_link))
b = 2.0
print('d = 2, beta = 2: exact tanh beta = %.5f; single flipped plaquettes 1 - 2 exp(-2 beta) = %.5f; link-flip formula 1 - 8 exp(-4 beta) = %.5f'
      % (np.tanh(b), 1 - 2 * np.exp(-2 * b), 1 - 8 * np.exp(-4 * b)))
# next order in three dimensions: vortex loops of 6 reversed plaquettes = self-avoiding 6-step polygons of the dual cubic lattice
steps = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
def walks(path, n):
    if len(path) == n + 1:
        return 1 if path[-1] == (0, 0, 0) else 0
    tot = 0
    for s in steps:
        q = tuple(a + b for a, b in zip(path[-1], s))
        if q == (0, 0, 0) and len(path) == n:
            tot += walks(path + [q], n)
        elif q not in path:
            tot += walks(path + [q], n)
    return tot
p4, p6 = walks([(0, 0, 0)], 4) // 8, walks([(0, 0, 0)], 6) // 12
print('polygons per site of the cubic lattice: length 4: %d, length 6: %d' % (p4, p6))
print('d = 3: <sigma_P> = 1 - 2 (4 p4/3 exp(-8 beta) + 6 p6/3 exp(-12 beta)) = 1 - %d exp(-8 beta) - %d exp(-12 beta)' % (2 * 4 * p4 // 3, 2 * 6 * p6 // 3))
for b in (1.0, 1.2):
    print('   beta = %.1f: 1 - 8 exp(-8 beta) = %.5f, with the second term %.5f' % (b, 1 - 8 * np.exp(-8 * b), 1 - 8 * np.exp(-8 * b) - 88 * np.exp(-12 * b)))
