# name: fa-jump-formula-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionDerivatives.jump_formula file "fa-jump-formula-test.sage"
# description: for the square wave piece g = -1, h = 1 at a = 0 and φ = (1 - x²)² on [-1, 1], Sage evaluates both sides (the jump 2 φ(0))
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
CASES = [(['fun _ => -1', '_', 'fun _ => 1', '_', 'phi', '_', '0', '-1', '1', '_', '_', '_', '_', '_', '_', '_', '_'], True, {'g': 'lambda t: -1', 'h': 'lambda t: 1', 'a': '0'}), (['fun x => x', '_', 'fun x => x ^ 2 + 1', '_', 'phi', '_', '0', '-1', '1', '_', '_', '_', '_', '_', '_', '_', '_'], False, {'g': 'lambda t: t', 'h': 'lambda t: t**2 + 1', 'a': '0'}), (['Real.sin', '_', 'Real.cos', '_', 'phi', '_', '1/2', '-1', '1', '_', '_', '_', '_', '_', '_', '_', '_'], False, {'g': 'lambda t: sin(t)', 'h': 'lambda t: cos(t)', 'a': '1/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('-(integrate(g(y)*diff((1 - y**2)**2, y), y, -1, a) + integrate(h(y)*diff((1 - y**2)**2, y), y, a, 1))', scope))
    R = side(eval('integrate(diff(g(y), y)*(1 - y**2)**2, y, -1, a) + integrate(diff(h(y), y)*(1 - y**2)**2, y, a, 1) + (h(a) - g(a))*(1 - a**2)**2', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
