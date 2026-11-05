# name: fa-lebesgue-constant-growth-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Convergence.LebesgueConstantGrowth file "fa-lebesgue-constant-growth-test.sage"
# description: Sage integrates |D_N| exactly between the consecutive zeros 2πj/(2N+1) of D_N (antiderivative t + 2Σ sin((n+1)t)/(n+1)) and compares with (8/π) log N in intervals
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
def absint(N):
    N = int(N)
    F = lambda t: t + 2 * sum(sin((m + 1) * t) / (m + 1) for m in range(N))
    zs = [-pi] + [2 * pi * j / (2 * N + 1) for j in range(-N, N + 1)] + [pi]
    return sum(abs(RI(F(zs[j + 1]) - F(zs[j]))) for j in range(len(zs) - 1))
instances = []
CASES = [(['1', 'le_refl 1'], True, {'N': '1'}), (['10', '_'], False, {'N': '10'}), (['100', '_'], False, {'N': '100'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('8/pi*log(N)', scope))
    R = side(eval('absint(N)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
