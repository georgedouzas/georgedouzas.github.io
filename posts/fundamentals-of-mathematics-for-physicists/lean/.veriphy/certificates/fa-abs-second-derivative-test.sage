# name: fa-abs-second-derivative-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionDerivatives.abs_second_derivative file "fa-abs-second-derivative-test.sage"
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
CASES = [(['fun x => (1 - x ^ 2) ^ 3', '_', '_', '1', '_', '_', '_', '_', '_', '_', '_', '_', '_'], True, {'p': 'lambda t: (1 - t**2)**3', 'c': '1'}), (['fun x => (4 - x ^ 2) ^ 3 * Real.cos x', '_', '_', '2', '_', '_', '_', '_', '_', '_', '_', '_', '_'], False, {'p': 'lambda t: (4 - t**2)**3*cos(t)', 'c': '2'}), (['fun x => (1 - x ^ 2) ^ 3 * (x + 3)', '_', '_', '1', '_', '_', '_', '_', '_', '_', '_', '_', '_'], False, {'p': 'lambda t: (1 - t**2)**3*(t + 3)', 'c': '1'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(-y*diff(p(y), y, 2), y, -c, 0) + integrate(y*diff(p(y), y, 2), y, 0, c)', scope))
    R = side(eval('2*p(0)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
