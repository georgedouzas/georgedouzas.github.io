# Exercise exr-spd-monopole-casimirs: levels of D^2 on the unit sphere with a monopole of charge M = 3 from Casimirs:
# E_j = ((2j+1)^2 - M^2)/4 on the components of U(1) charge (M - 1)/2 and (M + 1)/2 (j >= |charge|).
from collections import Counter
M = 3
levels = Counter()
for q in ((M - 1) / 2, (M + 1) / 2):
    for twoj in range(int(2 * abs(q)), 20, 2):
        j = twoj / 2
        levels[((2 * j + 1) ^ 2 - M ^ 2) / 4] += 2 * j + 1
for E in sorted(levels)[:5]:
    k = (-M + sqrt(M ^ 2 + 4 * E)) / 2
    print("E = %3s = k(k + |M|) with k = %s: multiplicity %3d, formula %s" % (E, k, levels[E], M if k == 0 else 2 * (M + 2 * k)))
