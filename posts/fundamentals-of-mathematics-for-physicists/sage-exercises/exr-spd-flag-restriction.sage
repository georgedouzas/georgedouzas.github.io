# Exercise exr-spd-flag-restriction: weights of SU(3) representations (p, pbar) as U(1) x U(1) charges (T, Y) in the
# normalisation of Dolan (2003), eq. (69): the fundamental 3 has (1, 1), (-1, 1), (0, -2).
from collections import Counter
W = WeylCharacterRing("A2")
def charges(p, pb):
    ch = W(p + pb, pb, 0)                      # highest weight in the epsilon basis of gl(3)
    out = Counter()
    for wt, mult in ch.weight_multiplicities().items():
        e = list(wt.to_vector()); m = sum(e) / 3; e = [x - m for x in e]
        out[(e[0] - e[1], e[0] + e[1] - 2 * e[2])] += mult      # T = 2 I_3, Y with the 3 at (1,1), (-1,1), (0,-2)
    return out
print("charges of the 3:", dict(charges(1, 0)))
for (p, pb) in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (3, 0), (2, 1)):
    c = charges(p, pb)
    print("(%d,%d), dim %2d: multiplicity of (2,0): %d, of (0,2): %d, of (0,0): %d" % (p, pb, W(p + pb, pb, 0).degree(), c[(2, 0)], c[(0, 2)], c[(0, 0)]))
