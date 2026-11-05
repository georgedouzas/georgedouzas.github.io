# name: fa-extremal-gaussian-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Uncertainty.extremal_gaussian file "fa-extremal-gaussian-test.sage"
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
instances = []
CASES = [(['-1', 'fun x => 3 * Real.exp (-x ^ 2 / 2)', '_', '0'], True, {'lam': '-1', 'C': '3', 'x': '0'}), (['-2', 'fun x => Real.exp (-x ^ 2)', '_', '1'], False, {'lam': '-2', 'C': '1', 'x': '1'}), (['1/2', 'fun x => -Real.exp (x ^ 2 / 4)', '_', '3'], False, {'lam': '1/2', 'C': '-1', 'x': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('C*exp(lam*x**2/2)', scope))
    R = side(eval('C*exp(lam*0**2/2)*exp(lam*x**2/2)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
