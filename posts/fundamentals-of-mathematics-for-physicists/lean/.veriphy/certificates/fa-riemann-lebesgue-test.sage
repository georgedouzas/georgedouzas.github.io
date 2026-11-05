# name: fa-riemann-lebesgue-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Convergence.riemann_lebesgue file "fa-riemann-lebesgue-test.sage"
# description: for the Gaussian f(v) = e^{-v²} Sage evaluates ∫ e^{-2πivw} f(v) dv = √π e^{-π²w²}, which tends to 0
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
y = var('y'); k = var('k'); i = var('i'); n = var('n'); z = var('z'); v = var('v')
instances = []
CASES = [(['fun v => Complex.exp (-(v : ℂ) ^ 2)'], True, {'w': '0'}), (['fun v => Complex.exp (-(v : ℂ) ^ 2)'], False, {'w': '1'}), (['fun v => Complex.exp (-(v : ℂ) ^ 2)'], False, {'w': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(exp(-2*pi*I*v*w)*exp(-v**2), v, -oo, oo)', scope))
    R = side(eval('sqrt(pi)*exp(-pi**2*w**2)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
