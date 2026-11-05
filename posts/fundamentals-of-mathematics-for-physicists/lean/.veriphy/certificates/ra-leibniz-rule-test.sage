# name: ra-leibniz-rule-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.leibniz_rule file "ra-leibniz-rule-test.sage"
# description: Sage differentiates F(a) = ∫_c^d f(x, a) dx and integrates ∂ₐ f: both sides agree
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
x, a = var('x a')
instances = []
for f, c, d, av, edge in [(exp(-a * x ** 2), 0, 1, 1, False), (sin(a * x), 0, pi, 2, False), (x ** 2 * a ** 3, -1, 1, QQ(1) / 2, False), (SR(0), 0, 1, 0, True)]:
    lhs = diff(integrate(f, x, c, d), a).subs(a=av); rhs = integrate(diff(f, a).subs(a=av), x, c, d)
    instances.append(O([f'(fun a x => {f})', f'(fun a x => {diff(f, a)})', str(c), str(d), str(av), '(by fun_prop)', '(by fun_prop)', '‹_›'], lhs, rhs, edge=edge))
