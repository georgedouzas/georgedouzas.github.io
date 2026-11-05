# name: fa-gibbs-overshoot-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.PartialSums.GibbsOvershoot file "fa-gibbs-overshoot-test.sage"
# description: Sage computes Si(π) with Arb balls and S_{2M}(π/2M) in 128-bit intervals; 1 means the four strict inequalities hold
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
RB = RealBallField(128)
G = RI(RB(2) / RB(pi) * RB(pi).Si())
def gibbs(M):
    M = int(M)
    S = sum(RI(2 * (1 - (-1) ** (n + 1))) / (RI(pi) * (n + 1)) * RI(sin((n + 1) * pi / (2 * M))) for n in range(2 * M))
    ok = (RI('1.1789') < G) and (G < RI('1.179')) and (S - G > 0) and (S - G < RI(1) / M ** 2)
    print('M', M, 'S', S, 'G', G)
    return 1 if ok else 0
instances = []
CASES = [(['1', 'le_refl 1'], True, {'M': '1'}), (['5', '(by norm_num)'], False, {'M': '5'}), (['100', '(by norm_num)'], False, {'M': '100'}), (['500', '(by norm_num)'], False, {'M': '500'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('gibbs(M)', scope))
    R = side(eval('1', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
