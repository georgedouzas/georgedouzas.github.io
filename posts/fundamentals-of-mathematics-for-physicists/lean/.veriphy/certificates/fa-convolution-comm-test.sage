# name: fa-convolution-comm-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.ConvolutionProduct.convolution_comm_lsmul file "fa-convolution-comm-test.sage"
# description: Sage evaluates (f ⋆ g)(x) and (g ⋆ f)(x) at x = 1
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
CASES = [(['fun x => Real.exp (-x ^ 2)', 'fun x => Real.exp (-x ^ 2)'], True, {'f': 'lambda t: exp(-t**2)', 'g': 'lambda t: exp(-t**2)'}), (['fun x => Real.exp (-x ^ 2)', 'fun x => x * Real.exp (-x ^ 2 / 2)'], False, {'f': 'lambda t: exp(-t**2)', 'g': 'lambda t: t*exp(-t**2/2)'}), (['fun x => Real.exp (-|x|)', 'fun x => Real.exp (-x ^ 2)'], False, {'f': 'lambda t: exp(-t**2/3)', 'g': 'lambda t: t**2*exp(-t**2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(f(1 - y)*g(y), y, -oo, oo)', scope))
    R = side(eval('integrate(g(1 - y)*f(y), y, -oo, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
