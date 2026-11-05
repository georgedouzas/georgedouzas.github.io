# name: fa-fejer-nonneg-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PartialSums.fejer_nonneg file "fa-fejer-nonneg-test.sage"
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
def D(n, t):
    return 1 + 2*sum(cos((j + 1)*t) for j in range(int(n)))
instances = []
CASES = [(['0', '0'], True, {'N': '0', 't': '0'}), (['3', '1'], False, {'N': '3', 't': '1'}), (['6', '3'], False, {'N': '6', 't': '3'}), (['4', '2 * π'], True, {'N': '4', 't': '2*pi'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('sum(D(n, t) for n in range(int(N) + 1))', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='>=', edge=edge))
    print(values, L, R)
