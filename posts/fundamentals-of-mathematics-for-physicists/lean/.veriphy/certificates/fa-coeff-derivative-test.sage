# name: fa-coeff-derivative-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.FourierCoefficients.coeff_derivative file "fa-coeff-derivative-test.sage"
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
y = var('y')
instances = []
CASES = [(['fun x => (Real.sin x ^ 2 : ℂ)', 'fun x => (2 * Real.sin x * Real.cos x : ℂ)', '_', '_', '_', '0'], True, {'f': 'lambda z: sin(z)**2', 'fp': 'lambda z: 2*sin(z)*cos(z)', 'n': '0'}), (['fun x => (Real.sin x ^ 2 : ℂ)', 'fun x => (2 * Real.sin x * Real.cos x : ℂ)', '_', '_', '_', '2'], False, {'f': 'lambda z: sin(z)**2', 'fp': 'lambda z: 2*sin(z)*cos(z)', 'n': '2'}), (['fun x => (Real.cos (3 * x) : ℂ)', 'fun x => (-3 * Real.sin (3 * x) : ℂ)', '_', '_', '_', '-3'], False, {'f': 'lambda z: cos(3*z)', 'fp': 'lambda z: -3*sin(3*z)', 'n': '-3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(fp(y)*exp(-n*y*I), y, 0, 2*pi)', scope))
    R = side(eval('n*I*integrate(f(y)*exp(-n*y*I), y, 0, 2*pi)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
