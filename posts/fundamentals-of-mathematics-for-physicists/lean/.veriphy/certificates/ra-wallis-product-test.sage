# name: ra-wallis-product-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Exercises.wallis_product file "ra-wallis-product-test.sage"
# description: (2k+1)/(2k+2) π/2 ≤ W_k ≤ π/2 for Wallis's partial products, by 128-bit intervals
SEED = 0  # the instances are listed, no sampling
R = RealIntervalField(128)
def B(b):
    return QQ(1 if bool(b) else 0)
RI = RealIntervalField(256)
def O(values, lhs, rhs, rel='==', edge=False):
    # symbolic values (π, e, logs, limits Sage computed) are compared by 256-bit intervals; rationals exactly
    if any(hasattr(v, 'parent') and v.parent() is SR for v in (lhs, rhs)):
        lhs, rhs = RI(lhs), RI(rhs)
    return oracle_instance(*values, lhs=lhs, rhs=rhs, relation=rel, edge=edge)
instances = []
for kk in [0, 1, 10, 100]:
    W = prod(QQ(2 * i + 2) / (2 * i + 1) * QQ(2 * i + 2) / (2 * i + 3) for i in range(kk))
    ok = R(2 * kk + 1) / (2 * kk + 2) * R(pi) / 2 <= R(W) and R(W) <= R(pi) / 2
    instances.append(O([str(kk)], B(ok), QQ(1), edge=bool(kk == 0)))
