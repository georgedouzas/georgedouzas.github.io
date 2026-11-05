# name: fa-plancherel-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Plancherel.plancherel file "fa-plancherel-test.sage"
# description: for Gaussians Sage computes ∫|f|² and ∫|𝓕 f|² (𝓕 in mathlib's normalisation)
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
CASES = [(['gaussianLp (fun x => Real.exp (-π * x ^ 2))'], True, {'f': 'lambda t: exp(-pi*t**2)', 'F': 'lambda w: exp(-pi*w**2)'}), (['gaussianLp (fun x => Real.exp (-x ^ 2))'], False, {'f': 'lambda t: exp(-t**2)', 'F': 'lambda w: sqrt(pi)*exp(-pi**2*w**2)'}), (['gaussianLp (fun x => Real.exp (-x ^ 2 / 2))'], False, {'f': 'lambda t: exp(-t**2/2)', 'F': 'lambda w: sqrt(2*pi)*exp(-2*pi**2*w**2)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('sqrt(integrate(abs(F(y))**2, y, -oo, oo))', scope))
    R = side(eval('sqrt(integrate(abs(f(y))**2, y, -oo, oo))', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
