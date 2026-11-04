# Exercise exr-vcr-cst-count: the field content of the three consistent sphere truncations of supergravity.
# Gauge fields: dim SO(n+1) for S^4, S^5, S^7. For N = 8 in four dimensions: 28 vectors and 70 scalars, the
# 70 = binomial(8, 4) splitting into 35 self-dual and 35 anti-self-dual four-forms of SO(8) (35 scalars, 35 pseudoscalars).
for n, D, d in [(4, 11, 7), (5, 10, 5), (7, 11, 4)]:
    print("S^%d reduction of D = %d to d = %d: SO(%d) with dim %d gauge fields" % (n, D, d, n + 1, binomial(n + 1, 2)))
print("binomial(8, 4) =", binomial(8, 4), "= 35 + 35; dim E7 - dim SU(8) =", 133 - 63)
# self-dual and anti-self-dual parts: eigenvalues of the Hodge star on 4-forms of R^8 (Euclidean): *^2 = 1
import itertools
idx = list(itertools.combinations(range(8), 4)); pos = {I: k for k, I in enumerate(idx)}
star = matrix(QQ, 70, 70)
for I_ in idx:
    J = tuple(sorted(set(range(8)) - set(I_)))
    sgn = Permutation([i + 1 for i in I_ + J]).sign()
    star[pos[J], pos[I_]] = sgn
ev = star.eigenvalues()
print("Hodge star on Lambda^4 R^8: eigenvalue +1 with multiplicity", ev.count(1), ", -1 with multiplicity", ev.count(-1))
