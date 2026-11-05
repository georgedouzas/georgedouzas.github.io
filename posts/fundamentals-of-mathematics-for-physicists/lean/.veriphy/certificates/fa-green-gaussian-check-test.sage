# name: fa-green-gaussian-check-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionDerivatives.GreenGaussianCheck file "fa-green-gaussian-check-test.sage"
# description: Sage integrates the even integrand on [0, 60] with Arb balls (rigorous) and bounds the tail by e^{-3000}
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
def gg():
    val = CB.integral(lambda x, _: -CB(1)/2*(-x).exp()*((4*x*x - 2)*(-x*x).exp() - (-x*x).exp()), 0, 60)
    return 2*RI(val.real())
instances = []
CASES = [(['0'], True, {'c': '0'}), (['1'], False, {'c': '1'}), (['3'], False, {'c': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('c*gg()', scope))
    R = side(eval('c', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
