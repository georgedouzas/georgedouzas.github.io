# name: fa-coulomb-delta-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionComputations.coulomb_delta file "fa-coulomb-delta-test.sage"
# description: Sage evaluates -4π(ε ∂_rφ + φ) at ε = 10⁻⁶ for φ = c e^{-r²}: within 10⁻¹⁰ (1 + |c|) of -4πc
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
instances = []
CASES = [(['0'], True, {'c': '0'}), (['1'], False, {'c': '1'}), (['5'], False, {'c': '5'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(-4*pi*(SR(10)**(-6)*(-2*SR(10)**(-6)*c*exp(-SR(10)**(-12))) + c*exp(-SR(10)**(-12))) + 4*pi*c)', scope))
    R = side(eval('SR(10)**(-10)*(1 + abs(c))', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
