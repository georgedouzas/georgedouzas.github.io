# name: fa-convolution-theorem-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.ConvolutionProduct.convolution_theorem file "fa-convolution-theorem-test.sage"
# description: for Gaussians Sage computes f ⋆ g, then 𝓕(f ⋆ g)(ξ) and 𝓕 f(ξ) 𝓕 g(ξ)
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
x_ = var('x_')
instances = []
CASES = [(['fun x => Complex.exp (-π * (x : ℂ) ^ 2)', 'fun x => Complex.exp (-π * (x : ℂ) ^ 2)', '_', '_', '0'], True, {'xi': '0', 'f': 'lambda t: exp(-pi*t**2)', 'g': 'lambda t: exp(-pi*t**2)'}), (['fun x => Complex.exp (-π * (x : ℂ) ^ 2)', 'fun x => Complex.exp (-(x : ℂ) ^ 2)', '_', '_', '1'], False, {'xi': '1', 'f': 'lambda t: exp(-pi*t**2)', 'g': 'lambda t: exp(-t**2)'}), (['fun x => Complex.exp (-(x : ℂ) ^ 2 / 2)', 'fun x => Complex.exp (-(x : ℂ) ^ 2)', '_', '_', '1/2'], False, {'xi': '1/2', 'f': 'lambda t: exp(-t**2/2)', 'g': 'lambda t: exp(-t**2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(exp(-2*pi*I*x_*xi)*integrate(f(y)*g(x_ - y), y, -oo, oo), x_, -oo, oo)', scope))
    R = side(eval('integrate(exp(-2*pi*I*y*xi)*f(y), y, -oo, oo)*integrate(exp(-2*pi*I*y*xi)*g(y), y, -oo, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
