# name: fa-poisson-summation-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Plancherel.poisson_summation file "fa-poisson-summation-test.sage"
# description: for f = e^{-πx²} (its own 𝓕) Sage sums |n| ≤ 30 in 128-bit intervals; the tails are below e^{-π·900}
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
def lhs_(x):
    return sum(RI(exp(-pi*(x + n)**2)) for n in range(-30, 31))
def rhs_(x):
    return sum(CI(exp(-pi*n**2)*exp(2*pi*I*n*x)) for n in range(-30, 31))
instances = []
CASES = [(['gaussianSchwartz', '0'], True, {'x': '0'}), (['gaussianSchwartz', '1/3'], False, {'x': '1/3'}), (['gaussianSchwartz', '5/2'], False, {'x': '5/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('lhs_(x)', scope))
    R = side(eval('rhs_(x)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
