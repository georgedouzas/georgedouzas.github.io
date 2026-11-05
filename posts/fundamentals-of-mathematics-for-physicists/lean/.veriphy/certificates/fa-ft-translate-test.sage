# name: fa-ft-translate-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.TheTransform.ft_translate file "fa-ft-translate-test.sage"
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
y = var('y'); k_ = var('k_'); x_ = var('x_'); assume(y, 'real')
instances = []
CASES = [(['fun x => Complex.exp (-(x : ℂ) ^ 2)', '0', '0'], True, {'f': 'lambda t: exp(-t**2)', 'a': '0', 'k': '0'}), (['fun x => Complex.exp (-(x : ℂ) ^ 2)', '3', '1'], False, {'f': 'lambda t: exp(-t**2)', 'a': '3', 'k': '1'}), (['fun x => Complex.exp (-(x : ℂ) ^ 2)', '-1/2', '2'], False, {'f': 'lambda t: exp(-t**2)', 'a': '-1/2', 'k': '2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(f(y - a)*exp(-k*y*I), y, -oo, oo)', scope))
    R = side(eval('exp(-k*a*I)*integrate(f(y)*exp(-k*y*I), y, -oo, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
