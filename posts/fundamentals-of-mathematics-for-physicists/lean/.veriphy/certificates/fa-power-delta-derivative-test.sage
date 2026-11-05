# name: fa-power-delta-derivative-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionDerivatives.PowerDeltaDerivative file "fa-power-delta-derivative-test.sage"
# description: Sage evaluates (x^n φ)^{(n)}(0) for φ = cos, eˣ, 1/(1 + x) and compares with n! φ(0)
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
z_ = var('z_')
instances = []
CASES = [(['0', 'Real.cos', '_'], True, {'n': '0', 'p': 'lambda t: cos(t)'}), (['3', 'Real.exp', '_'], False, {'n': '3', 'p': 'lambda t: exp(t)'}), (['5', 'fun x => 1 / (1 + x ^ 2)', '_'], False, {'n': '5', 'p': 'lambda t: 1/(1 + t**2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('diff(z_**n*p(z_), z_, int(n)).subs(z_=0)', scope))
    R = side(eval('factorial(n)*p(0)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
