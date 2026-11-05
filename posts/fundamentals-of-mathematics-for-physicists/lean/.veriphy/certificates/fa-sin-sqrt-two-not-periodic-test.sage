# name: fa-sin-sqrt-two-not-periodic-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeriodicFunctions.sin_sqrt_two_not_periodic file "fa-sin-sqrt-two-not-periodic-test.sage"
# description: the implication at T: Sage evaluates 1 when T = 0 or when the hypothesis fails at x = 0 (|g(T) - g(0)| > 0, by intervals)
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
def g(z):
    return sin(z) + sin(sqrt(2)*z)
def imp(T):
    return 1 if bool(T == 0) else (1 if RI(abs(g(T) - g(0))) > 0 else 0)
instances = []
CASES = [(['0', '_'], True, {'T': '0'}), (['2 * π', '_'], False, {'T': '2*pi'}), (['1', '_'], False, {'T': '1'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('imp(T)', scope))
    R = side(eval('1', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
