# name: fa-dipole-moments-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionComputations.DipoleMoments file "fa-dipole-moments-test.sage"
# description: Sage integrates the four moments with Arb balls over [-40ε, 40ε] (tails below e^{-1500}) and evaluates the sum of the four deviations
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
CB = ComplexBallField(128)
def dev(e):
    e = CB(e)
    sp = CB(pi).sqrt()
    g = lambda x: (-(x*x)/(e*e)).exp()/(e*sp)
    gp = lambda x: -2*x/(e*e)*g(x)
    d = lambda x: (g(x + e) - g(x - e))/(2*e)
    I = lambda f: CB.integral(lambda x, _: f(x), -40*e, 40*e)
    t = [I(lambda x: gp(x)*x**3) + 3*e*e/2, I(lambda x: d(x)*x**3) + 3*e*e/2 + e*e, I(lambda x: d(x)*x) + 1, I(lambda x: gp(x)*x) + 1]
    return sum(RI(z.real()).abs() + RI(z.imag()).abs() for z in t)
instances = []
CASES = [(['1', 'one_pos'], True, {'e': '1'}), (['1/2', '_'], False, {'e': '1/2'}), (['2', '_'], False, {'e': '2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('dev(e)', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
