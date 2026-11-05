# name: fa-l1-le-seminorm-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TestFunctions.l1_le_seminorm file "fa-l1-le-seminorm-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
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
CASES = [(['fun x => 1 / (1 + x ^ 2)', '1', '_', '_'], True, {'g': 'lambda t: 1/(1 + t**2)', 'C': '1'}), (['fun x => Real.exp (-x ^ 2)', '1', '_', '_'], False, {'g': 'lambda t: exp(-t**2)', 'C': '1'}), (['fun x => x * Real.exp (-x ^ 2)', '1', '_', '_'], False, {'g': 'lambda t: t*exp(-t**2)', 'C': '1'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = SR(eval('2*integrate(abs(g(y)), y, 0, oo)', scope))
    R = SR(eval('pi*C', scope))
    d = (L - R).simplify_full().canonicalize_radical()
    if bool(d == 0):
        R = L  # proven equal symbolically
    else:
        L, R = num(L), num(R)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
    continue
    L = side(eval('2*integrate(abs(g(y)), y, 0, oo)', scope))
    R = side(eval('pi*C', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
