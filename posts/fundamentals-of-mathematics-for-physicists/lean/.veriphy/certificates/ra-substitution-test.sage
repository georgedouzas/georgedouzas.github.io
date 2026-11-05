# name: ra-substitution-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.substitution file "ra-substitution-test.sage"
# description: both sides of ∫ f(g(x)) g'(x) dx = ∫_{g(a)}^{g(b)} f(u) du computed exactly by Sage
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
x, u = var('x u')
instances = []
for f, g, a, b, edge in [(sqrt(1 + u ** 2) * 0 + u ** 2, sin(x), 0, pi / 2, False), (exp(u), x ** 2, 0, 1, False), (cos(u), 2 * x, 0, pi / 4, False), (u, x, 0, 1, True)]:
    lhs = integrate(f.subs(u=g) * diff(g, x), x, a, b); rhs = integrate(f, u, g.subs(x=a), g.subs(x=b))
    instances.append(O([str(f), str(g), str(diff(g, x)), str(a), str(b), '‹_›', '‹_›', '‹_›'], lhs, rhs, edge=edge))
