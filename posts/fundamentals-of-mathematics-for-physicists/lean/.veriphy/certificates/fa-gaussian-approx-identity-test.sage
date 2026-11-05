# name: fa-gaussian-approx-identity-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Inversion.gaussian_approx_identity file "fa-gaussian-approx-identity-test.sage"
# description: for f = e^{-w²} at v = 0 Sage evaluates the Gaussian average at c = 10³, 10⁴ in closed form; it is within 1/(π² c) of f(0) = 1
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
c_ = var('c_')
instances = []
CASES = [(['fun w => Complex.exp (-(w : ℂ) ^ 2)', '_', '0', '_'], True, {'c': '1000'}), (['fun w => Complex.exp (-(w : ℂ) ^ 2)', '_', '0', '_'], False, {'c': '10000'}), (['fun w => Complex.exp (-(w : ℂ) ^ 2)', '_', '0', '_'], False, {'c': '100000'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(integrate(sqrt(pi*c)*exp(-pi**2*c*y**2)*exp(-y**2), y, -oo, oo) - 1)', scope))
    R = side(eval('1/(pi**2*c)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
