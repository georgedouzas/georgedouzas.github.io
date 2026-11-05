# name: fa-l2-convergence-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Convergence.l2_convergence file "fa-l2-convergence-test.sage"
# description: for f(x) = x on [0, 2π) Sage evaluates (1/2π)∫|f|² - Σ_{|n|≤N}|c_n|² against the tail 2Σ_{n>N} 1/n², which tends to 0
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
CASES = [(['2 * π', '_', 'sawtoothLp'], True, {'N': '0'}), (['2 * π', '_', 'sawtoothLp'], False, {'N': '3'}), (['2 * π', '_', 'sawtoothLp'], False, {'N': '10'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('1/(2*pi)*integrate(y**2, y, 0, 2*pi) - (pi**2 + 2*sum(1/SR(j)**2 for j in range(1, int(N) + 1)))', scope))
    R = side(eval('2*sum(1/n**2, n, N + 1, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
