# name: ra-step-integrable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.step_integrable file "ra-step-integrable-test.sage"
# description: Sage integrates step functions exactly (u(c - a) + v(b - c))
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
for a, b, c, u, v, edge in [(0, 1, QQ(1) / 2, 1, 2, False), (-1, 3, 0, -2, 5, False), (0, 0, 0, 0, 0, True)]:
    val = integrate(u, x, a, c) + integrate(v, x, c, b)
    instances.append(O([str(a), str(b), str(c), str(u), str(v)], B(val in QQ), QQ(1), edge=edge))
