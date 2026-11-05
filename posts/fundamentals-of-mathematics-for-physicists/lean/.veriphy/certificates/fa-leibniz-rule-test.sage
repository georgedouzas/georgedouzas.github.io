# name: fa-leibniz-rule-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionDerivatives.leibniz_rule file "fa-leibniz-rule-test.sage"
# description: for the regular distribution T_f = ∫ f · on [-1, 1] with ψ = x² + 1 and φ = (1 - x²)², Sage evaluates both sides
SEED = 0
RI = RealIntervalField(128)
CI = ComplexIntervalField(128)
def num(e):
    try:
        return RI(e)
    except (TypeError, ValueError):
        pass
    e = SR(e)
    for f in (lambda t: t, lambda t: t.simplify_full(), lambda t: t.canonicalize_radical()):
        try:
            return RI(f(e))
        except (TypeError, ValueError):
            pass
    try:
        return CI(e)
    except (TypeError, ValueError):
        return RI(e.n(prec=256))
def side(e):
    v = num(e)
    return v
y = var('y'); z_ = var('z_'); assume(y, 'real')
instances = []
CASES = [(['regular 1', 'fun x => x ^ 2 + 1', 'fun x => (1 - x ^ 2) ^ 2', '_', '_'], True, {'f': 'lambda t: 1'}), (['regular cos', 'fun x => x ^ 2 + 1', 'fun x => (1 - x ^ 2) ^ 2', '_', '_'], False, {'f': 'lambda t: cos(t)'}), (['regular exp', 'fun x => x ^ 2 + 1', 'fun x => (1 - x ^ 2) ^ 2', '_', '_'], False, {'f': 'lambda t: exp(t)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('-integrate(f(y)*(y**2 + 1)*diff((1 - y**2)**2, y), y, -1, 1)', scope))
    R = side(eval('integrate(f(y)*2*y*(1 - y**2)**2, y, -1, 1) - integrate(f(y)*diff((y**2 + 1)*(1 - y**2)**2, y), y, -1, 1)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
