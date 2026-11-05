# name: fa-cos-mul-not-tendsto-zero-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Convergence.cos_mul_not_tendsto_zero file "fa-cos-mul-not-tendsto-zero-test.sage"
# description: Sage evaluates max |cos n x| over n ∈ [1000, 2000] in intervals: it stays ≥ 1/2, so the terms do not tend to 0
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
def mx(x):
    return max((RI(abs(cos(m * x))) for m in range(1000, 2001)), key=lambda r: r.lower())
instances = []
CASES = [(['0'], True, {'x': '0'}), (['1'], False, {'x': '1'}), (['π / 2'], False, {'x': 'pi/2'}), (['1/1000'], False, {'x': '1/1000'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('mx(x)', scope))
    R = side(eval('1/2', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='>=', edge=edge))
    print(values, L, R)
