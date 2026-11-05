# name: fa-transform-self-adjoint-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionTransform.TransformSelfAdjoint file "fa-transform-self-adjoint-test.sage"
# description: for Gaussians Sage computes ∫ f φ̂ and ∫ f̂ φ
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
k_ = var('k_')
instances = []
CASES = [(['fun x => Complex.exp (-(x : ℂ) ^ 2)', 'fun x => Complex.exp (-(x : ℂ) ^ 2)', '_', '_'], True, {'f': 'lambda t: exp(-t**2)', 'p': 'lambda t: exp(-t**2)'}), (['fun x => Complex.exp (-(x : ℂ) ^ 2)', 'fun x => Complex.exp (-(x : ℂ) ^ 2 / 2)', '_', '_'], False, {'f': 'lambda t: exp(-t**2)', 'p': 'lambda t: exp(-t**2/2)'}), (['fun x => Complex.exp (-(x - 1 : ℂ) ^ 2)', 'fun x => Complex.exp (-(x : ℂ) ^ 2 / 3)', '_', '_'], False, {'f': 'lambda t: exp(-(t - 1)**2)', 'p': 'lambda t: exp(-t**2/3)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(f(y)*integrate(p(k_)*exp(-I*y*k_), k_, -oo, oo), y, -oo, oo)', scope))
    R = side(eval('integrate(integrate(f(k_)*exp(-I*y*k_), k_, -oo, oo)*p(y), y, -oo, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
