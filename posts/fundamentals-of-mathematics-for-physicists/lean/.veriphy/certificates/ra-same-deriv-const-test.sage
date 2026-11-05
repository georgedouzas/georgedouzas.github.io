# name: ra-same-deriv-const-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.same_deriv_const file "ra-same-deriv-const-test.sage"
# description: f - g is constant when Sage's f' = g' (sin² vs -cos², x² + 3 vs x²)
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
for f, g, edge in [(sin(x) ** 2, -cos(x) ** 2, False), (x ** 2 + 3, x ** 2, False), (exp(x), exp(x), True)]:
    ok = bool((diff(f, x) - diff(g, x)).simplify_full() == 0) and bool(diff((f - g).simplify_full(), x) == 0)
    instances.append(O([str(f), str(g), '‹_›', '‹_›', '‹_›'], B(ok), QQ(1), edge=edge))
