# name: fa-exp-not-integrable-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TransformComputations.exp_not_integrable file "fa-exp-not-integrable-test.sage"
# description: Sage evaluates ∫_{-100}^{100} e^{-a|x|} ≥ 200 for a ≤ 0: the integral over the line diverges
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
y = var('y'); k_ = var('k_'); x_ = var('x_'); assume(y, 'real')
instances = []
CASES = [(['0', 'le_refl 0'], True, {'a': '0'}), (['-1', '_'], False, {'a': '-1'}), (['-1/2', '_'], False, {'a': '-1/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('2*integrate(exp(-a*y), y, 0, 100)', scope))
    R = side(eval('200', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='>=', edge=edge))
    print(values, L, R)
