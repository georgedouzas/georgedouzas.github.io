# Exercise exr-spd-rp3-spectrum: the Dirac spectrum of RP^3 = SU(2)/{+-1} from that of SU(2), keeping the summands
# V_j (x) V_j^* with integer j, compared with Trautman's spectrum {-1/2 +- (2l + 2)}, multiplicity 2 binom(|l| + 1/2, |l| - 3/2).
from collections import Counter
spec = Counter()
for twoj in range(0, 12, 2):
    j = twoj / 2
    spec[2 * j + 3/2] += (2 * j + 1) * (2 * j + 2)
    if j > 0: spec[-(2 * j + 1/2)] += (2 * j + 1) * 2 * j
for lam in sorted(spec, key=abs)[:8]:
    a = abs(lam)
    print("lambda = %5s: multiplicity %3d, Trautman: %3d" % (lam, spec[lam], 2 * binomial(a + 1/2, a - 3/2)))
print("Trautman's eigenvalues -1/2 +- (2l + 2) for l = 0..3:", sorted([-1/2 + (2 * l + 2) for l in range(4)] + [-1/2 - (2 * l + 2) for l in range(4)], key=abs))
