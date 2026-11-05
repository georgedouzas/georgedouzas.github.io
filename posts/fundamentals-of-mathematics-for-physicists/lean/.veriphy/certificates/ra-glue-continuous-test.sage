# name: ra-glue-continuous-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.glue_continuous file "ra-glue-continuous-test.sage"
# description: Sage's one-sided limits at a and the value decide continuity; d = 0 exactly when they agree
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
for a, d, edge in [(1, 0, True), (1, 1, False), (2, -3, False), (0, 0, False)]:
    left = limit(x ** 2, x=a, dir='-'); right = limit(2 * a * x - a ** 2 + d, x=a, dir='+'); val = 2 * a * a - a ** 2 + d
    instances.append(O([str(a), str(d)], B(bool(left == right == val)), B(d == 0), edge=edge))
