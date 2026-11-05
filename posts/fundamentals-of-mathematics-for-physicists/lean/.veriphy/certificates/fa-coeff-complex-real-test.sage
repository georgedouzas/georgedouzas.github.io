# name: fa-coeff-complex-real-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.FourierCoefficients.coeff_complex_real file "fa-coeff-complex-real-test.sage"
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
CASES = [(['fun x => x ^ 2', '(by fun_prop)', '0'], True, {'f': 'lambda z: z**2', 'n': '0'}), (['fun x => x ^ 2', '(by fun_prop)', '1'], False, {'f': 'lambda z: z**2', 'n': '1'}), (['fun x => Real.exp (Real.cos x)', '(by fun_prop)', '2'], False, {'f': 'lambda z: z*sin(z) + cos(2*z)', 'n': '2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(f(y)*exp(-n*y*I), y, 0, 2*pi)', scope))
    R = side(eval('integrate(f(y)*cos(n*y), y, 0, 2*pi) - I*integrate(f(y)*sin(n*y), y, 0, 2*pi)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
