# name: fa-pointwise-of-summable-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Convergence.pointwise_of_summable file "fa-pointwise-of-summable-test.sage"
# description: for f = cos on the circle (c_{±1} = 1/2) Sage evaluates the series c_1 e^{ix} + c_{-1} e^{-ix} against f(x)
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
CASES = [(['2 * π', '⟨Real.two_pi_pos⟩', 'cosOnCircle', '_', '0'], True, {'x': '0'}), (['2 * π', '_', 'cosOnCircle', '_', '1'], False, {'x': '1'}), (['2 * π', '_', 'cosOnCircle', '_', '5/2'], False, {'x': '5/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('(exp(I*x) + exp(-I*x))/2', scope))
    R = side(eval('cos(x)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
