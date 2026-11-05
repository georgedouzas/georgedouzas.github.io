# name: fa-mollifier-approx-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TestFunctions.mollifier_approx file "fa-mollifier-approx-test.sage"
# description: for f = sin and the triangle kernel of mass 1 on [-δ, δ] Sage evaluates |(f * φ)(x) - f(x)| ≤ sup_{|y|≤δ}|f(x - y) - f(x)| ≤ δ
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
CASES = [(['Real.sin', 'tri', '_', '_', '_', '1/10', '1/10', '0', '_', '_', '_', '_'], True, {'d': '1/10', 'x': '0'}), (['Real.sin', 'tri', '_', '_', '_', '1/2', '1/2', '1', '_', '_', '_', '_'], False, {'d': '1/2', 'x': '1'}), (['Real.sin', 'tri', '_', '_', '_', '1', '1', '3', '_', '_', '_', '_'], False, {'d': '1', 'x': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(integrate(sin(x - y)*(d - abs(y))/d**2, y, -d, d) - sin(x))', scope))
    R = side(eval('d', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
