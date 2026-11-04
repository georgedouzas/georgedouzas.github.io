# Exercise exr-lgt-plaq-z2-orbits: Z2 links on the periodic 2 x 2 lattice: configurations, gauge orbits and plaquettes.
from itertools import product
L = 2
links = [(mu, x, y) for mu in (0, 1) for x in range(L) for y in range(L)]
def plaqs(s):
    return tuple(s[(0, x, y)] * s[(1, (x + 1) % L, y)] * s[(0, x, (y + 1) % L)] * s[(1, x, y)] for x in range(L) for y in range(L))
def transform(s, g):
    return {(mu, x, y): g[(x, y)] * s[(mu, x, y)] * g[((x + 1) % L, y) if mu == 0 else (x, (y + 1) % L)] for (mu, x, y) in links}
sites = [(x, y) for x in range(L) for y in range(L)]
configs = [dict(zip(links, v)) for v in product([1, -1], repeat=len(links))]
gauges = [dict(zip(sites, v)) for v in product([1, -1], repeat=len(sites))]
orbits = set()
for s in configs:
    orbits.add(min(tuple(transform(s, g)[l] for l in links) for g in gauges))
print('configurations:', len(configs), '; gauge transformations:', len(gauges), '; orbits:', len(orbits))
print('plaquettes invariant under every gauge transformation:', all(plaqs(transform(s, g)) == plaqs(s) for s in configs for g in gauges))
print('trivially acting transformations:', sum(all(transform(configs[5], g)[l] == configs[5][l] for l in links) for g in gauges))
print('product of the four plaquettes is always 1:', all(prod(plaqs(s)) == 1 for s in configs))
