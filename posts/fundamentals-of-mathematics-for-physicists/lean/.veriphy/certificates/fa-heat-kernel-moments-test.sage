# name: fa-heat-kernel-moments-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionComputations.heat_kernel_moments file "fa-heat-kernel-moments-test.sage"
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
CASES = [(['1', '1', 'one_pos', 'one_pos'], True, {'kap': '1', 't': '1'}), (['2', '1/10', '_', '_'], False, {'kap': '2', 't': '1/10'}), (['1/3', '5', '_', '_'], False, {'kap': '1/3', 't': '5'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(integrate(exp(-y**2/(4*kap*t))/sqrt(4*pi*kap*t)*cos(y), y, -oo, oo) - exp(-kap*t)) + abs(integrate(exp(-y**2/(4*kap*t))/sqrt(4*pi*kap*t)*(y**2 + 1), y, -oo, oo) - 1 - 2*kap*t)', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
