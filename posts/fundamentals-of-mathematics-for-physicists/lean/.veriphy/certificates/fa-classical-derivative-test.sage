# name: fa-classical-derivative-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.DistributionDerivatives.classical_derivative file "fa-classical-derivative-test.sage"
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
y = var('y'); z_ = var('z_'); assume(y, 'real')
instances = []
CASES = [(['Real.sin', 'Real.cos', 'fun x => x * (1 - x)', 'fun x => 1 - 2 * x', '0', '1', '_', '_', '_', '_', '_', '_'], True, {'f': 'lambda t: sin(t)', 'p': 'lambda t: t*(1 - t)', 'a': '0', 'b': '1'}), (['fun x => x ^ 3', '_', 'fun x => Real.sin x', '_', '0', 'π', '_', '_', '_', '_', '_', '_'], False, {'f': 'lambda t: t**3', 'p': 'lambda t: sin(t)', 'a': '0', 'b': 'pi'}), (['Real.exp', 'Real.exp', 'fun x => x ^ 2 - 4', '_', '-2', '2', '_', '_', '_', '_', '_', '_'], False, {'f': 'lambda t: exp(t)', 'p': 'lambda t: t**2 - 4', 'a': '-2', 'b': '2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('integrate(f(y)*diff(p(y), y), y, a, b)', scope))
    R = side(eval('-integrate(diff(f(y), y)*p(y), y, a, b)', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
