# Exercise exr-csf-charges: charges q = (h1 - h2)/2 of the weights of SU(3) representations under tau' = (i/2) diag(1,-1,0),
# where a weight is the triple (h1, h2, h3) of eigenvalues of diag(1,0,0), diag(0,1,0), diag(0,0,1); weights from Sage's
# character ring of A2 in the ambient basis.
A2 = WeylCharacterRing("A2", style="coroots")
from collections import Counter
for hw, name in (((1, 0), "3"), ((2, 0), "6"), ((1, 1), "8"), ((3, 0), "10"), ((2, 1), "15")):
    ch = A2(hw)
    qs = Counter()
    for w, mult in ch.weight_multiplicities().items():
        h = [QQ(c) for c in w.to_vector()]
        qs[(h[0] - h[1]) / 2] += mult
    print("%-3s dim %2d: charges %s ; |q| = 1/2 components: q = +1/2: %d, q = -1/2: %d" % (name, ch.degree(), sorted(qs.items()), qs[QQ(1) / 2], qs[-QQ(1) / 2]))
