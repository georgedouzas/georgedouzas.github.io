# name: ra-sign-one-sided-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.sign_one_sided file "ra-sign-one-sided-test.sage"
# description: Sage's one-sided limits of |x - a|/(x - a) at a are ±1
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
instances = []
for a in [0, 1, -2]:
    r = limit(abs(x - a) / (x - a), x=a, dir='+'); l = limit(abs(x - a) / (x - a), x=a, dir='-')
    instances.append(O([str(a)], B(r == 1 and l == -1), QQ(1), edge=bool(a == 0)))
