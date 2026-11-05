# name: ra-glue-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.glue_deriv file "ra-glue-deriv-test.sage"
# description: Sage's one-sided derivatives of the tangent gluing at a are both 2a
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
x, h = var('x h')
instances = []
for a in [0, 1, -2]:
    left = limit(((a + h) ** 2 - a ** 2) / h, h=0, dir='-'); right = limit((2 * a * (a + h) - a ** 2 - a ** 2) / h, h=0, dir='+')
    instances.append(O([str(a)], B(bool(left == right == 2 * a)), QQ(1), edge=bool(a == 0)))
