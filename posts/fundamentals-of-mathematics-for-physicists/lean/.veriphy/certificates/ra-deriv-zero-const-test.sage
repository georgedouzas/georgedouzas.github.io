# name: ra-deriv-zero-const-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.deriv_zero_const file "ra-deriv-zero-const-test.sage"
# description: a function with zero derivative (Sage) takes equal values at sample points
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
for f, edge in [(SR(3), True), (sin(x) ** 2 + cos(x) ** 2, False), (log(exp(2 + 0 * x)), False)]:
    ok = bool(diff(f, x).simplify_full() == 0) and bool((f.subs(x=0) - f.subs(x=5)).simplify_full() == 0)
    instances.append(O([str(f), '‹_›', '‹_›', '0', '5'], B(ok), QQ(1), edge=edge))
