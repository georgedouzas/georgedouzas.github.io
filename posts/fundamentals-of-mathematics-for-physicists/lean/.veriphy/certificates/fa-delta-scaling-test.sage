# name: fa-delta-scaling-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Distributions.delta_scaling file "fa-delta-scaling-test.sage"
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
CASES = [(['fun x => Real.exp (-x ^ 2)', 'Real.cos', '1', 'one_ne_zero'], True, {'g': 'lambda t: exp(-t**2)', 'p': 'lambda t: cos(t)', 'a': '1'}), (['fun x => Real.exp (-x ^ 2)', 'Real.cos', '2', '_'], False, {'g': 'lambda t: exp(-t**2)', 'p': 'lambda t: cos(t)', 'a': '2'}), (['fun x => Real.exp (-x ^ 2)', 'fun x => x ^ 2 + 1', '-3', '_'], False, {'g': 'lambda t: exp(-t**2)', 'p': 'lambda t: t**2 + 1', 'a': '-3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(g(a*y)*p(y), y, -oo, oo)', scope))
    R = side(eval('abs(1/a)*integrate(g(y)*p(y/a), y, -oo, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
