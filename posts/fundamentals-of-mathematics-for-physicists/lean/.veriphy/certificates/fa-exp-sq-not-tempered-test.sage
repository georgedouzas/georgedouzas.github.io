# name: fa-exp-sq-not-tempered-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Distributions.exp_sq_not_tempered file "fa-exp-sq-not-tempered-test.sage"
# description: Sage evaluates ∫_{-R}^{R} |c e^{x²} e^{-x²/2}| ≥ 2R|c| at R = 100: the integral over the line diverges
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
CASES = [(['1', 'one_ne_zero'], True, {'c': '1'}), (['-2', '_'], False, {'c': '-2'}), (['1/10', '_'], False, {'c': '1/10'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(c)*2*integrate(exp(y**2/2), y, 0, 100)', scope))
    R = side(eval('200*abs(c)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='>=', edge=edge))
    print(values, L, R)
