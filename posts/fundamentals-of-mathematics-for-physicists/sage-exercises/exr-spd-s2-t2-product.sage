# Exercise exr-spd-s2-t2-product: S^2 x T^2 (unit radii) with a monopole M on S^2 and twists alpha on T^2. Eigenvalues of the
# square: k(k + |M|) + |m + alpha|^2. Zero modes: k = 0 (|M| modes of chirality sign M on S^2) times the constant spinors of
# T^2 (one of each chirality, only for alpha = 0).
from collections import Counter
M = 2
for alpha in ((0, 0), (1/2, 0)):
    c = Counter()
    for k in range(0, 4):
        mult_s2 = abs(M) if k == 0 else 2 * (abs(M) + 2 * k)
        for m1 in range(-3, 4):
            for m2 in range(-3, 4):
                c[k * (k + abs(M)) + (m1 + alpha[0]) ^ 2 + (m2 + alpha[1]) ^ 2] += mult_s2 * 2
    low = sorted(c)[:5]
    print("alpha = %s: lowest eigenvalues of the square (multiplicity): %s" % (alpha, ", ".join("%s (%d)" % (v, c[v]) for v in low)))
print("zero modes for alpha = (0,0): %d of total chirality +1 (S^2 + times T^2 +) and %d of chirality -1 (S^2 + times T^2 -): index 0" % (abs(M), abs(M)))
