# name: fa-test-is-schwartz-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TestFunctions.test_is_schwartz file "fa-test-is-schwartz-test.sage"
# description: for the bump Sage evaluates x^m φ^{(n)}(x) at points of (-1, 1): finite values, the function vanishing outside
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
z_ = var('z_')
instances = []
CASES = [(['bump', '_', '_', '0', '0'], True, {'m': '0', 'n': '0'}), (['bump', '_', '_', '2', '1'], False, {'m': '2', 'n': '1'}), (['bump', '_', '_', '3', '2'], False, {'m': '3', 'n': '2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(RI((z_**m*diff(exp(-1/(1 - z_**2)), z_, int(n))).subs(z_=SR(1)/2)))', scope))
    R = side(eval('10**6', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
