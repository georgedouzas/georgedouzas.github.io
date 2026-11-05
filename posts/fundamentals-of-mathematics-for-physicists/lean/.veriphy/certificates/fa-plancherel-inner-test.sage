# name: fa-plancherel-inner-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Plancherel.plancherel_inner file "fa-plancherel-inner-test.sage"
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
CASES = [(['gaussianLp', 'gaussianLp'], True, {'f': 'lambda t: exp(-pi*t**2)', 'F': 'lambda w: exp(-pi*w**2)', 'g': 'lambda t: exp(-pi*t**2)', 'G': 'lambda w: exp(-pi*w**2)'}), (['gaussianLp', "gaussianLp'"], False, {'f': 'lambda t: exp(-pi*t**2)', 'F': 'lambda w: exp(-pi*w**2)', 'g': 'lambda t: exp(-t**2)', 'G': 'lambda w: sqrt(pi)*exp(-pi**2*w**2)'}), (["gaussianLp'", "gaussianLp''"], False, {'f': 'lambda t: exp(-t**2)', 'F': 'lambda w: sqrt(pi)*exp(-pi**2*w**2)', 'g': 'lambda t: exp(-t**2/2)', 'G': 'lambda w: sqrt(2*pi)*exp(-2*pi**2*w**2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(F(y)*G(y), y, -oo, oo)', scope))
    R = side(eval('integrate(f(y)*g(y), y, -oo, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
