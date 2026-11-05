# name: fa-exp-abs-not-tempered-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Distributions.exp_abs_not_tempered file "fa-exp-abs-not-tempered-test.sage"
# description: Sage integrates e^{|x| - √(1+x²)} over [0, 100] with Arb balls (rigorous): ∫_{-R}^{R} |c| e^{|x| - √(1+x²)} > 2R|c|e^{-1} at R = 100
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
CB = ComplexBallField(128)
def tail_int():
    return RI(CB.integral(lambda t, _: (t - (1 + t*t).sqrt()).exp(), 0, 100).real())
instances = []
CASES = [(['1', 'one_ne_zero'], True, {'c': '1'}), (['-2', '_'], False, {'c': '-2'}), (['1/10', '_'], False, {'c': '1/10'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(c)*2*tail_int()', scope))
    R = side(eval('200*abs(c)*exp(-1)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='>', edge=edge))
    print(values, L, R)
