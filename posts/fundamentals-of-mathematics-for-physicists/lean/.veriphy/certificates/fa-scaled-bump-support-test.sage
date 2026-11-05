# name: fa-scaled-bump-support-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TestFunctions.scaled_bump_support file "fa-scaled-bump-support-test.sage"
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
def bump(x):
    x = SR(x)
    return 0 if bool(abs(x) >= 1) else exp(-1/(1 - x**2))
instances = []
CASES = [(['0', '1', '1', 'one_pos', '_'], True, {'a': '0', 'e': '1', 'x': '1'}), (['2', '1/2', '3', '_', '_'], False, {'a': '2', 'e': '1/2', 'x': '3'}), (['-1', '3', '5', '_', '_'], False, {'a': '-1', 'e': '3', 'x': '5'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('bump((x - a)/e)', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
