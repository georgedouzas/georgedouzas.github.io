# name: fa-yukawa-radial-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TransformComputations.yukawa_radial file "fa-yukawa-radial-test.sage"
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
y = var('y'); k_ = var('k_'); x_ = var('x_'); assume(y, 'real')
instances = []
CASES = [(['1', '1', 'one_pos', 'one_ne_zero'], True, {'m': '1', 'k': '1'}), (['2', '3', '_', '_'], False, {'m': '2', 'k': '3'}), (['1/2', '-1', '_', '_'], False, {'m': '1/2', 'k': '-1'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('1/k*integrate(exp(-m*y)*sin(k*y), y, 0, oo)', scope))
    R = side(eval('1/(k**2 + m**2)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
