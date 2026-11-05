# name: ra-integral-monotone-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_monotone file "ra-integral-monotone-test.sage"
# description: f ≤ g gives ∫ f ≤ ∫ g: Sage's exact integrals (sin ≤ 1, x² ≤ x on [0, 1])
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
for f, g, a, b, edge in [(sin(x), SR(1), 0, pi, False), (x ** 2, x, 0, 1, False), (SR(0), SR(0), 0, 1, True)]:
    instances.append(O([str(f), str(g), str(a), str(b), '(by norm_num)', '(by fun_prop)', '(by fun_prop)', '‹_›'],
                       integrate(f, x, a, b), integrate(g, x, a, b), rel='<=', edge=edge))
