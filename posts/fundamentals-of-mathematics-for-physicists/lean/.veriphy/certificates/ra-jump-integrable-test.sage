# name: ra-jump-integrable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.jump_integrable file "ra-jump-integrable-test.sage"
# description: Sage integrates bounded functions with one jump (piecewise) exactly
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
for left, right, c, a, b, edge in [(x ** 2, 2 * x, 1, 0, 2, False), (sin(x), cos(x), 0, -1, 1, False), (SR(0), SR(1), 0, 0, 1, True)]:
    val = integrate(left, x, a, c) + integrate(right, x, c, b)
    instances.append(O([f'(fun x => if x < {c} then {left} else {right})', str(a), str(b), str(c), '4', '‹_›', '‹_›'],
                       B(RI(val).absolute_diameter() < 1), QQ(1), edge=edge))
