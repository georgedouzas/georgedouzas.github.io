# name: fa-real-coeff-conj-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.FourierCoefficients.real_coeff_conj file "fa-real-coeff-conj-test.sage"
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
CASES = [(['fun x => x', '0'], True, {'f': 'lambda z: z', 'n': '0'}), (['fun x => x', '1'], False, {'f': 'lambda z: z', 'n': '1'}), (['fun x => x ^ 2 + Real.cos x', '3'], False, {'f': 'lambda z: z**2 + cos(z)', 'n': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('conjugate(integrate(f(y)*exp(-n*y*I), y, 0, 2*pi))', scope))
    R = side(eval('integrate(f(y)*exp(n*y*I), y, 0, 2*pi)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
