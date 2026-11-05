# name: fa-pv-integrand-bound-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PrincipalValue.pv_integrand_bound file "fa-pv-integrand-bound-test.sage"
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
instances = []
CASES = [(['Real.sin', '1', '_', '_', '1', '_'], True, {'p': 'lambda t: sin(t)', 'M': '1', 'x': '1'}), (['Real.sin', '1', '_', '_', '1/10', '_'], False, {'p': 'lambda t: sin(t)', 'M': '1', 'x': '1/10'}), (['fun x => Real.cos (2 * x)', '2', '_', '_', '3', '_'], False, {'p': 'lambda t: cos(2*t)', 'M': '2', 'x': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(p(x) - p(-x))', scope))
    R = side(eval('2*M*x', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
