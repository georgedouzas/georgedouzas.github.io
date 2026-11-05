# name: fa-shrinking-integral-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Distributions.shrinking_integral file "fa-shrinking-integral-test.sage"
# description: for locally integrable f (|x|^{-1/2}, log|x|, 1) Sage evaluates ∫_{-ε}^{ε}|f| at ε = 10⁻⁶: below 10⁻²
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
CASES = [(['fun x => 1', '_'], True, {'f': 'lambda t: 1'}), (['fun x => |x| ^ (-1/2 : ℝ)', '_'], False, {'f': 'lambda t: 1/sqrt(t)'}), (['fun x => Real.log |x|', '_'], False, {'f': 'lambda t: -log(t)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('2*integrate(f(y), y, 0, SR(10)**(-6))', scope))
    R = side(eval('SR(10)**(-2)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
