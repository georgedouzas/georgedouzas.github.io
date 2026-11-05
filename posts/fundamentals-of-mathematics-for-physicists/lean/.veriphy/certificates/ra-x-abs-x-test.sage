# name: ra-x-abs-x-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.x_abs_x file "ra-x-abs-x-test.sage"
# description: Sage's one-sided difference quotients: (x|x|)' = 2|x|, and 2|x| has one-sided derivatives ±2 at 0
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
for v in [0, 1, -2]:
    d = limit(((v + h) * abs(v + h) - v * abs(v)) / h, h=0)
    corner = limit(2 * abs(h) / h, h=0, dir='+') == 2 and limit(2 * abs(h) / h, h=0, dir='-') == -2
    instances.append(O([str(v)], B(bool(d == 2 * abs(v)) and corner), QQ(1), edge=bool(v == 0)))
