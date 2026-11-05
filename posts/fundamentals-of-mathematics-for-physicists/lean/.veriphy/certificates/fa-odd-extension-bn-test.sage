# name: fa-odd-extension-bn-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.SeriesComputations.odd_extension_bn file "fa-odd-extension-bn-test.sage"
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
y = var('y'); k = var('k'); i = var('i'); n = var('n'); z = var('z'); v = var('v')
instances = []
CASES = [(['fun x => x', 'continuous_id', '1'], True, {'f': 'lambda t: t', 'n': '1'}), (['fun x => x ^ 2', '_', '2'], False, {'f': 'lambda t: t**2', 'n': '2'}), (['Real.cos', '_', '3/2'], False, {'f': 'lambda t: cos(t)', 'n': '3/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('1/pi*(integrate(-f(-y)*sin(n*y), y, -pi, 0) + integrate(f(y)*sin(n*y), y, 0, pi))', scope))
    R = side(eval('2/pi*integrate(f(y)*sin(n*y), y, 0, pi)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
