# name: fa-mollifier-support-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TestFunctions.mollifier_support file "fa-mollifier-support-test.sage"
# description: Sage integrates f(y) φ(x - y) for box-shaped f and φ supported in [-R, R] and [-r, r], at |x| > R + r
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
CASES = [(['box1', 'box1', '1', '1', '_', '_', '3', '_'], True, {'x': '3'}), (['box1', 'box1', '1', '1', '_', '_', '-5/2', '_'], False, {'x': '-5/2'}), (['box1', 'box1', '1', '1', '_', '_', '10', '_'], False, {'x': '10'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(integrate(1, y, max(-1, x - 1), min(1, x + 1)) if bool(abs(x) <= 2) else 0, z_, 0, 1)', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
