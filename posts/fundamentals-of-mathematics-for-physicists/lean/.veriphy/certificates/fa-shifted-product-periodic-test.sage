# name: fa-shifted-product-periodic-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PeriodicFunctions.shifted_product_periodic file "fa-shifted-product-periodic-test.sage"
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
CASES = [(['fun x => x', 'fun x => Real.cos x + Real.sin (2 * x)', '_', '0'], True, {'f': 'lambda z: z', 'g': 'lambda z: cos(z) + sin(2*z)', 't': '0'}), (['fun x => x', 'fun x => Real.cos x + Real.sin (2 * x)', '_', '1'], False, {'f': 'lambda z: z', 'g': 'lambda z: cos(z) + sin(2*z)', 't': '1'}), (['fun x => x ^ 2', 'fun x => Real.cos (3 * x)', '_', '1/3'], False, {'f': 'lambda z: z**2', 'g': 'lambda z: cos(3*z)', 't': '1/3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(f(y)*g(y - (t + 2*pi)), y, 0, 2*pi)', scope))
    R = side(eval('integrate(f(y)*g(y - t), y, 0, 2*pi)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
