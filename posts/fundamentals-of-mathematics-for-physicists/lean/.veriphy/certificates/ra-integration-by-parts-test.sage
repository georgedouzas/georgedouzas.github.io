# name: ra-integration-by-parts-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integration_by_parts file "ra-integration-by-parts-test.sage"
# description: both sides of ∫ f g' = fg| - ∫ f' g computed exactly by Sage
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
for f, g, a, b, edge in [(x, exp(x), 0, 1, False), (log(x + 2), x, 0, 1, False), (x ** 2, sin(x), 0, pi, False), (SR(1), x, 0, 1, True)]:
    lhs = integrate(f * diff(g, x), x, a, b); rhs = (f * g).subs(x=b) - (f * g).subs(x=a) - integrate(diff(f, x) * g, x, a, b)
    instances.append(O([str(f), str(g), str(diff(f, x)), str(diff(g, x)), str(a), str(b), '‹_›', '‹_›', '‹_›', '‹_›'], lhs, rhs, edge=edge))
