# name: fa-inversion-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Inversion.inversion file "fa-inversion-test.sage"
# description: for Gaussians Sage computes 𝓕⁻(𝓕 f) at x = 1/2 by the two integrals and compares with f(1/2)
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
w_ = var('w_'); v_ = var('v_')
instances = []
CASES = [(['fun x => Complex.exp (-π * (x : ℂ) ^ 2)', '_', '_', '_'], True, {'f': 'lambda t: exp(-pi*t**2)'}), (['fun x => Complex.exp (-(x : ℂ) ^ 2)', '_', '_', '_'], False, {'f': 'lambda t: exp(-t**2)'}), (['fun x => Complex.exp (-(x - 1 : ℂ) ^ 2 / 2)', '_', '_', '_'], False, {'f': 'lambda t: exp(-(t - 1)**2/2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(exp(2*pi*I*w_*(1/2))*integrate(exp(-2*pi*I*v_*w_)*f(v_), v_, -oo, oo), w_, -oo, oo)', scope))
    R = side(eval('f(1/2)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
