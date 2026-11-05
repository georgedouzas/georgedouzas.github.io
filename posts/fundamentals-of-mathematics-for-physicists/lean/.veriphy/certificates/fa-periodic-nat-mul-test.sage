# name: fa-periodic-nat-mul-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeriodicFunctions.periodic_nat_mul file "fa-periodic-nat-mul-test.sage"
# description: Sage evaluates f(x + kT) - f(x) for periodic sample functions (sin, cos, sin 3x) in 128-bit intervals
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
CASES = [(['fun y => Real.sin y', '2 * π', 'Real.sin_periodic', '0', '1/3'], True, {'f': 'lambda y: sin(y)', 'T': '2*pi', 'k': '0', 'x': '1/3'}), (['fun y => Real.cos y', '2 * π', 'Real.cos_periodic', '3', '1/2'], False, {'f': 'lambda y: cos(y)', 'T': '2*pi', 'k': '3', 'x': '1/2'}), (['fun y => Real.sin (3 * y)', '2 * π / 3', 'fun y => by sorry', '2', '1/5'], False, {'f': 'lambda y: sin(3*y)', 'T': '2*pi/3', 'k': '2', 'x': '1/5'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('f(x + k*T)', scope))
    R = side(eval('f(x)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
