# name: fa-theta-symmetric-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.SphereLaplacian.theta_symmetric file "fa-theta-symmetric-test.sage"
# description: for f = cos θ (f' = -sin θ) and g = cos² θ, sin θ Sage evaluates both sides of the integration by parts
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
z_ = var('z_'); y = var('y'); assume(y, 'real')
instances = []
CASES = [(['fun t => -Real.sin t', 'fun t => 1', 'fun t => 0', 'fun t => -(2 * Real.sin t * Real.cos t)', '_', '_', '_', '_', '_'], True, {'g': 'lambda t: 1', 'fp': 'lambda t: -sin(t)'}), (['fun t => -Real.sin t', 'fun t => Real.cos t ^ 2', 'fun t => -(2 * Real.cos t * Real.sin t)', 'fun t => -(2 * Real.sin t * Real.cos t)', '_', '_', '_', '_', '_'], False, {'g': 'lambda t: cos(t)**2', 'fp': 'lambda t: -sin(t)'}), (['fun t => 3 * Real.cos t * Real.sin t', 'Real.sin', 'Real.cos', 'fun t => -3 * Real.sin t ^ 3 + 6 * Real.sin t * Real.cos t ^ 2', '_', '_', '_', '_', '_'], False, {'g': 'lambda t: sin(t)', 'fp': 'lambda t: 3*cos(t)*sin(t)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(g(y)*diff(sin(y)*fp(y), y), y, 0, pi)', scope))
    R = side(eval('-integrate(diff(g(y), y)*sin(y)*fp(y), y, 0, pi)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
