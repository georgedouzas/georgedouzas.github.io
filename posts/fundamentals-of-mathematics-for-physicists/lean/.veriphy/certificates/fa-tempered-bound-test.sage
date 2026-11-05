# name: fa-tempered-bound-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Distributions.tempered_bound file "fa-tempered-bound-test.sage"
# description: for f = 1 + |x| and φ = e^{-x²} with N = 2, Sage evaluates both sides
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
y = var('y'); z_ = var('z_'); assume(y, 'real')
instances = []
CASES = [(['fun x => 1', 'fun x => Real.exp (-x ^ 2)', '2', 'C', '_', '_', '_'], True, {'N': '2', 'f': 'lambda t: 1', 'p': 'lambda t: exp(-t**2)'}), (['fun x => x ^ 2', 'fun x => Real.exp (-x ^ 2)', '2', 'C', '_', '_', '_'], False, {'N': '2', 'f': 'lambda t: t**2', 'p': 'lambda t: exp(-t**2)'}), (['fun x => 1 + |x|', 'fun x => Real.exp (-x ^ 2 / 2)', '2', 'C', '_', '_', '_'], False, {'N': '2', 'f': 'lambda t: 1 + abs(t)', 'p': 'lambda t: exp(-t**2/2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('2*integrate(abs(f(y)*p(y)), y, 0, oo)', scope))
    R = side(eval('RI(find_local_maximum((1 + z_**2)**N*p(z_), 0, 3)[0]).upper()*2*integrate(abs(f(y))/(1 + y**2)**N, y, 0, oo)*1.0001', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
