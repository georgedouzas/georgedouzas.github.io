# name: fa-rescale-two-pi-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeriodicFunctions.rescale_two_pi file "fa-rescale-two-pi-test.sage"
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
CASES = [(['fun y => Real.sin (2 * π * y / 3)', '3', '_', '0'], True, {'f': 'lambda y: sin(2*pi*y/3)', 'T': '3', 'x': '0'}), (['fun y => Real.sin (2 * π * y / 3)', '3', '_', '1/2'], False, {'f': 'lambda y: sin(2*pi*y/3)', 'T': '3', 'x': '1/2'}), (['fun y => Real.cos (2 * π * y / 5) ^ 2', '5', '_', '7'], False, {'f': 'lambda y: cos(2*pi*y/5)**2', 'T': '5', 'x': '7'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('f(T*(x + 2*pi)/(2*pi))', scope))
    R = side(eval('f(T*x/(2*pi))', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
