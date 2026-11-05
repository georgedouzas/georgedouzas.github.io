# name: fa-sech-transform-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TransformComputations.SechTransform file "fa-sech-transform-test.sage"
# description: Sage integrates sech x e^{-ikx} over [-60, 60] with Arb balls (rigorous) and adds the tail bound 4e^{-60}
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
def sechft(k):
    kk = CB(k)
    val = CB.integral(lambda x, _: (1/x.cosh())*(-CB(0, 1)*kk*x).exp(), -60, 60)
    r = RI(val.real())
    tail = RI(4)*RI(exp(-60))
    return RI(r.lower() - tail.upper(), r.upper() + tail.upper())
instances = []
CASES = [(['0'], True, {'k': '0'}), (['1'], False, {'k': '1'}), (['-2'], False, {'k': '-2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('sechft(k)', scope))
    R = side(eval('pi/cosh(pi*k/2)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
