# name: fa-square-wave-dirichlet-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Convergence.SquareWaveDirichlet file "fa-square-wave-dirichlet-test.sage"
# description: Sage evaluates |S_N(x) - 1| at N = 2000 in intervals and checks it against Abel's tail bound 4/(π (N+1) sin x)
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
def SN(N, x):
    return sum(RI(4) / (RI(pi) * (2 * j + 1)) * RI(sin((2 * j + 1) * x)) for j in range(N // 2))
instances = []
CASES = [(['π / 2', '_', '_'], True, {'x': 'pi/2'}), (['1', '_', '_'], False, {'x': '1'}), (['1/10', '_', '_'], False, {'x': '1/10'}), (['3', '_', '_'], False, {'x': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(SN(2000, x) - 1)', scope))
    R = side(eval('4/(pi*2001*sin(x))', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
