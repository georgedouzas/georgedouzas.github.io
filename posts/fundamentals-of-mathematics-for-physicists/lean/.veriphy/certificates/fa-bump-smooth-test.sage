# name: fa-bump-smooth-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TestFunctions.bump_smooth file "fa-bump-smooth-test.sage"
# description: Sage evaluates the derivatives of exp(-1/(1-u²)), u = (x - a)/ε, near the edge of the support: they tend to 0, matching the zero extension
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
def bump(x):
    x = SR(x)
    return 0 if bool(abs(x) >= 1) else exp(-1/(1 - x**2))
instances = []
CASES = [(['0', '1'], True, {'a': '0', 'e': '1'}), (['2', '1/2'], False, {'a': '2', 'e': '1/2'}), (['-1', '3'], False, {'a': '-1', 'e': '3'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('abs(RI(diff(exp(-1/(1 - ((z_ - a)/e)**2)), z_, 3).subs(z_=a + e*(1 - SR(1)/10**3))))', scope))
    R = side(eval('SR(10)**(-100)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
