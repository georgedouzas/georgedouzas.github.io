# name: fa-poisson-gaussian-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Plancherel.poisson_gaussian file "fa-poisson-gaussian-test.sage"
# description: Sage sums |n| ≤ 40 in 128-bit intervals (tails below e^{-σ²·800})
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
def L_(s):
    return sum(RI(exp(-2*pi**2*n**2/s**2)) for n in range(-40, 41))
def R_(s):
    return RI(s/sqrt(2*pi))*sum(RI(exp(-s**2*n**2/2)) for n in range(-40, 41))
instances = []
CASES = [(['1', 'one_pos'], True, {'s': '1'}), (['2', '_'], False, {'s': '2'}), (['7/2', '_'], False, {'s': '7/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('L_(s)', scope))
    R = side(eval('R_(s)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
