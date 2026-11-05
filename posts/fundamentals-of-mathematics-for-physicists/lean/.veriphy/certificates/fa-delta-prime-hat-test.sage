# name: fa-delta-prime-hat-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionTransform.delta_prime_hat file "fa-delta-prime-hat-test.sage"
# description: for φ = e^{-πx²} (𝓕 φ = e^{-πw²}) Sage evaluates (𝓕 φ)'(0) = 0 and 𝓕(-2πix φ)(0) = 0; for φ = x e^{-πx²} both are -2πi·(1/(2π))·... compared in closed form
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
w_ = var('w_')
instances = []
CASES = [(['fun x => Complex.exp (-π * (x : ℂ) ^ 2)', '_', '_'], True, {'p': 'lambda t: exp(-pi*t**2)'}), (['fun x => (x : ℂ) * Complex.exp (-π * (x : ℂ) ^ 2)', '_', '_'], False, {'p': 'lambda t: t*exp(-pi*t**2)'}), (['fun x => Complex.exp (-π * (x - 1 : ℂ) ^ 2)', '_', '_'], False, {'p': 'lambda t: exp(-pi*(t - 1)**2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('diff(integrate(exp(-2*pi*I*y*w_)*p(y), y, -oo, oo), w_).subs(w_=0)', scope))
    R = side(eval('integrate(-2*pi*I*y*p(y), y, -oo, oo)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
