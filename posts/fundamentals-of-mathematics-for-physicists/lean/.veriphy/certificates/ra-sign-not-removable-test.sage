# name: ra-sign-not-removable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sign_not_removable file "ra-sign-not-removable-test.sage"
# description: the one-sided limits ±1 of |x|/x differ, so no value v at 0 makes it continuous
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
x = var('x')
r = limit(abs(x) / x, x=0, dir='+'); l = limit(abs(x) / x, x=0, dir='-')
instances = [O([str(v)], B(r != l), QQ(1), edge=bool(v == 0)) for v in [0, 1, -1]]
