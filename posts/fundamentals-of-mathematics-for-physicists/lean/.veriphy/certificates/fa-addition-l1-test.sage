# name: fa-addition-l1-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.SphericalHarmonics.addition_l1 file "fa-addition-l1-test.sage"
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
CASES = [(['0', '0', '0', '0'], True, {'a': '0', 'b': '0', 'c': '0', 'd': '0'}), (['1', '2', '3', '4'], False, {'a': '1', 'b': '2', 'c': '3', 'd': '4'}), (['1/2', '-1', '2', '1/3'], False, {'a': '1/2', 'b': '-1', 'c': '2', 'd': '1/3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('3/(4*pi)*cos(a)*cos(c) + 3/(8*pi)*sin(a)*sin(c)*(exp((b - d)*I) + exp(-(b - d)*I))', scope))
    R = side(eval('3/(4*pi)*(cos(a)*cos(c) + sin(a)*sin(c)*cos(b - d))', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
