# name: fa-radial-operator-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.SphereLaplacian.radial_operator file "fa-radial-operator-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
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
z_ = var('z_'); y = var('y'); assume(y, 'real')
instances = []
CASES = [(['fun r => r ^ 2', 'fun r => 2 * r', 'fun r => 2', '1', 'one_ne_zero', '_'], True, {'f': 'lambda r: r**2', 'r': '1'}), (['Real.exp', 'Real.exp', 'Real.exp', '2', '_', '_'], False, {'f': 'lambda r: exp(r)', 'r': '2'}), (['fun r => 1 / r', '_', '_', '-1/2', '_', '_'], False, {'f': 'lambda r: 1/r', 'r': '-1/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('diff(z_**2*diff(f(z_), z_), z_).subs(z_=r)/r**2', scope))
    R = side(eval('diff(f(z_), z_, 2).subs(z_=r) + 2/r*diff(f(z_), z_).subs(z_=r)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
