# Exercise exr-cff-z3-projection: Z_3 phases on the ten-dimensional gaugino of E_8 on T^6/Z_3. Spinor: the rotation
# 2 pi (1/3, 1/3, -2/3) acts on the Weyl weights (+-1/2)^3 (even number of minus signs) by exp(2 pi i v.h). Gauge: the
# standard embedding acts by omega on the 3 of SU(3) in E_8 -> SU(3) x E_6; without embedding it acts trivially.
import itertools
v = vector(QQ, (QQ(1) / 3, QQ(1) / 3, -QQ(2) / 3))
spin = [vector(QQ, h) / 2 for h in itertools.product((1, -1), repeat=3) if prod(h) == 1]
fr = lambda x: x - floor(x)
sph = [fr(v.dot_product(h)) for h in spin]                 # phases as fractions of a full turn
print("spinor phases (units of 2 pi):", sph)
gauge = [("(8,1)", 8, 0, "adjoint of SU(3)"), ("(1,78)", 78, 0, "78"), ("(3,27)", 3 * 27, QQ(1) / 3, "27"), ("(3b,27b)", 3 * 27, QQ(2) / 3, "27bar")]
for label, twisted in (("(a) standard embedding", True), ("(b) no gauge twist", False)):
    count = {}
    for name, dim, gph, rep in gauge:
        g = gph if twisted else 0
        n = sum(1 for p in sph if fr(p + g) == 0)
        if n: count[name] = n * dim
    print(label, ": invariant fermion components by (SU(3), E_6) representation:", count)
