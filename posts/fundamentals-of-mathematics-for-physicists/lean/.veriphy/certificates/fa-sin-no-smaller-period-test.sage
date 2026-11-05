# name: fa-sin-no-smaller-period-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_no_smaller_period file "fa-sin-no-smaller-period-test.sage"
# description: Sage evaluates |sin(π/2 + T) - sin(π/2)| > 0 for T in (0, 2π), the edge case T = 157/25 just below 2π
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
CASES = [(['157/25', '(by norm_num)', '(by linarith [Real.pi_gt_d2])'], True, {'T': '157/25'}), (['1', '(by norm_num)', '_'], False, {'T': '1'}), (['3', '(by norm_num)', '_'], False, {'T': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(sin(pi/2 + T) - sin(pi/2))', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='>', edge=edge))
    print(values, L, R)
