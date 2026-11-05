# name: fa-heat-semigroup-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.ConvolutionProduct.heat_semigroup file "fa-heat-semigroup-test.sage"
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
CASES = [(['1', '1', '0', 'one_pos', 'one_pos'], True, {'e': '1', 'h': '1', 'x': '0'}), (['2', '3', '1', '_', '_'], False, {'e': '2', 'h': '3', 'x': '1'}), (['1/2', '1/4', '-2', '_', '_'], False, {'e': '1/2', 'h': '1/4', 'x': '-2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(exp(-(x - y)**2/(2*e))/sqrt(2*pi*e)*exp(-y**2/(2*h))/sqrt(2*pi*h), y, -oo, oo)', scope))
    R = side(eval('exp(-x**2/(2*(e + h)))/sqrt(2*pi*(e + h))', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
