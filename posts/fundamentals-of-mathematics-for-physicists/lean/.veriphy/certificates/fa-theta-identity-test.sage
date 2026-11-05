# name: fa-theta-identity-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.ChapterExercises.ThetaIdentity file "fa-theta-identity-test.sage"
# description: Sage sums both series in 128-bit intervals with |n|, |k| ≤ 60 (tails below e^{-36} at t = 1/10 and e^{-3600}... at t = 1)
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
def lhs_(t, x):
    return RI(1/(2*pi))*sum(RI(exp(-n**2*t)*cos(n*x)) for n in range(-60, 61))
def rhs_(t, x):
    return sum(RI(exp(-(x - 2*pi*k)**2/(4*t))/sqrt(4*pi*t)) for k in range(-60, 61))
instances = []
CASES = [(['1', '0', 'one_pos'], True, {'t': '1', 'x': '0'}), (['1/10', '1', '_'], False, {'t': '1/10', 'x': '1'}), (['1', '5/2', '_'], False, {'t': '1', 'x': '5/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('lhs_(t, x)', scope))
    R = side(eval('rhs_(t, x)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
