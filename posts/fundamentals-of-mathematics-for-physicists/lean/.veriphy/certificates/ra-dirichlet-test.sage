# name: ra-dirichlet-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.dirichlet file "ra-dirichlet-test.sage"
# description: every interval contains a rational (inf 0... sup 1) and an irrational; the rationals have measure 0
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
for c, d, edge in [(0, 1, True), (QQ(1) / 3, QQ(1) / 2, False), (-2, sqrt(2), False)]:
    q = (QQ(c) + QQ(floor(d * 10 ** 6)) / 10 ** 6) / 2 if c in QQ else None   # a rational inside (c, d)
    irr = (RI(c) + RI(d)) / 2 + RI(sqrt(2)) / 10 ** 9   # an irrational point inside
    ok = q is not None and RI(c) < RI(q) < RI(d) and RI(c) < irr < RI(d)
    instances.append(O([str(c), str(d), '(by norm_num)'], B(ok), QQ(1), edge=edge))
