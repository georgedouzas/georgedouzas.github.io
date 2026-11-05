# name: fa-pv-symmetric-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PrincipalValue.pv_symmetric file "fa-pv-symmetric-test.sage"
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
CASES = [(['Real.cos', '1/10', '2', '_', '_'], True, {'p': 'lambda t: cos(t)', 'e': '1/10', 'c': '2'}), (['Real.exp', '1/2', '3', '_', '_'], False, {'p': 'lambda t: exp(t)', 'e': '1/2', 'c': '3'}), (['fun x => x ^ 2 + x + 1', '1', '4', '_', '_'], False, {'p': 'lambda t: t**2 + t + 1', 'e': '1', 'c': '4'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(p(y)/y, y, -c, -e) + integrate(p(y)/y, y, e, c)', scope))
    R = side(eval('integrate((p(y) - p(-y))/y, y, e, c)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
