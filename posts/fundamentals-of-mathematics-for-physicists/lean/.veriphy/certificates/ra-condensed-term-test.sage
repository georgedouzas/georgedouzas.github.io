# name: ra-condensed-term-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Series.condensed_term file "ra-condensed-term-test.sage"
# description: 2^m (2^m)^(-p) = (2^(1-p))^m exactly, and 2^(1-p) < 1 iff p > 1
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
for p, m, edge in [(1, 0, True), (2, 3, False), (QQ(1) / 2, 4, False), (3, 2, False)]:
    lhs = 2 ** m * (1 / (2 ** m) ** SR(p)); rhs = (2 ** (1 - SR(p))) ** m
    instances.append(O([str(p), str(m)], B(bool((lhs - rhs).simplify_full() == 0) and (bool(2 ** (1 - SR(p)) < 1) == (p > 1))), QQ(1), edge=edge))
