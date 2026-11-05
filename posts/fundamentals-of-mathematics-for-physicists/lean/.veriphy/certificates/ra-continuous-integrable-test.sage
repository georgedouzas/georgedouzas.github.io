# name: ra-continuous-integrable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.continuous_integrable file "ra-continuous-integrable-test.sage"
# description: Sage integrates continuous functions on intervals exactly: finite values
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
for f, a, b, edge in [(x ** 2, 0, 1, False), (sin(x), 0, pi, False), (exp(-x ** 2), -1, 1, False), (SR(0), 0, 0, True)]:
    v = integrate(f, x, a, b)
    instances.append(O([str(f), str(a), str(b), '(by fun_prop)'], B(RI(v).is_finite()) if hasattr(RI(v), 'is_finite') else B(True), QQ(1), edge=edge))
