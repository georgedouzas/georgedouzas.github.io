# name: ra-no-rat-lub-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.no_rat_lub file "ra-no-rat-lub-test.sage"
# description: p is not a perfect square, so √p is not rational and no rational is the least upper bound; p bounds the set
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
for p, edge in [(2, True), (3, False), (5, False), (7, False)]:
    assert not QQ(p).is_square()
    instances.append(O([str(p), '(by norm_num)'], B(sqrt(QQ(p)) in QQ), QQ(0), edge=edge))
