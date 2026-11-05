# name: fa-gaussian-complex-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TheTransform.gaussian_complex file "fa-gaussian-complex-test.sage"
# description: Sage integrates e^{-a x²} over [-8, 8] with Arb balls (rigorous) and widens by the tail bound e^{-64}
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
CB = ComplexBallField(64)
def gc(a):
    aa = CB(a)
    val = CB.integral(lambda x, _: (-aa*x*x).exp(), -8, 8)
    t = RI(exp(-64))
    return CI(RI(val.real().lower() - t.upper(), val.real().upper() + t.upper()), RI(val.imag().lower() - t.upper(), val.imag().upper() + t.upper()))
def gcr(a):
    z = CB(pi)/CB(a)
    w = z.sqrt()
    return CI(RI(w.real().lower(), w.real().upper()), RI(w.imag().lower(), w.imag().upper()))
instances = []
CASES = [(['1', '1', '_', '_'], True, {'a': '1'}), (['2', '1/2', '_', '_'], False, {'a': '2'}), (['1 + Complex.I', '(1 - Complex.I) / 2', '_', '_'], False, {'a': '1 + I'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('gc(a)', scope))
    R = side(eval('gcr(a)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
