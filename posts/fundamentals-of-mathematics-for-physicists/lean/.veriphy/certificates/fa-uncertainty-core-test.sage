# name: fa-uncertainty-core-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Uncertainty.uncertainty_core file "fa-uncertainty-core-test.sage"
# description: in ℂ² with ‖f‖² = -2 Re⟨X, D⟩ Sage evaluates ‖f‖² ≤ 2‖X‖‖D‖; the edge case attains equality
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
def nm(v):
    return sqrt(sum(SR(c)**2 for c in v))
instances = []
CASES = [(['EuclideanSpace ℂ (Fin 2)', '_', '_', '![1, 1]', '![1, 0]', '![-1, 0]', '_'], True, {'X': 'lambda: [1, 0]', 'D': 'lambda: [-1, 0]', 'f': 'lambda: [1, 1]'}), (['EuclideanSpace ℂ (Fin 2)', '_', '_', '![1, 1]', '![1, 1]', '![-1, 0]', '_'], False, {'X': 'lambda: [1, 1]', 'D': 'lambda: [-1, 0]', 'f': 'lambda: [1, 1]'}), (['EuclideanSpace ℂ (Fin 2)', '_', '_', '![2, 0]', '![2, 1]', '![-1, 0]', '_'], False, {'X': 'lambda: [2, 1]', 'D': 'lambda: [-1, 0]', 'f': 'lambda: [2, 0]'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = SR(eval('nm(f())**2', scope))
    R = SR(eval('2*nm(X())*nm(D())', scope))
    d = (L - R).simplify_full().canonicalize_radical()
    if bool(d == 0):
        R = L  # proven equal symbolically
    else:
        L, R = num(L), num(R)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
    continue
    L = side(eval('nm(f())**2', scope))
    R = side(eval('2*nm(X())*nm(D())', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
