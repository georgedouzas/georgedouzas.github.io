# name: ra-sin-nat-lt-one-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sin_nat_lt_one file "ra-sin-nat-lt-one-test.sage"
# description: sin n < 1 by 128-bit intervals, also at n = 33 where sin n = 0.99991
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
instances = [O([str(n)], R(sin(n)), R(1), rel='<', edge=bool(n == 0)) for n in [0, 1, 2, 33, 344]]
