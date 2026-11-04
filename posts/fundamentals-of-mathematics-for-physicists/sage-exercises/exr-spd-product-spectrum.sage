# Exercise exr-spd-product-spectrum: the smallest eigenvalues of D^2 = (k + 1)^2 + |p|^2 on S^2 x T^2 (unit radii), with
# multiplicity 8(k + 1) per pair (k, p), for the periodic and the antiperiodic torus.
from collections import Counter
for name, al in (("periodic T^2", (0, 0)), ("antiperiodic T^2", (1/2, 1/2))):
    c = Counter()
    for k in range(0, 5):
        for m1 in range(-4, 5):
            for m2 in range(-4, 5):
                c[(k + 1) ^ 2 + (m1 + al[0]) ^ 2 + (m2 + al[1]) ^ 2] += 8 * (k + 1)
    low = sorted(c)[:5]
    print(name.ljust(17), ", ".join("%s (x%d)" % (q, c[q]) for q in low), "| smallest:", low[0])
