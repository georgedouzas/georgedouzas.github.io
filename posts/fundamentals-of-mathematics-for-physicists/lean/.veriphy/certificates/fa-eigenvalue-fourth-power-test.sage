# name: fa-eigenvalue-fourth-power-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Inversion.eigenvalue_fourth_power file "fa-eigenvalue-fourth-power-test.sage"
# description: the eigenvalues ±√(2π), ±i√(2π) of the Hermite functions satisfy λ⁴ = (2π)²
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
instances = []
CASES = [(['fourierOp', '_', 'gaussian', 'Real.sqrt (2 * π)', '_', '_'], True, {'lam': 'sqrt(2*pi)'}), (['fourierOp', '_', 'hermite1', '-Complex.I * Real.sqrt (2 * π)', '_', '_'], False, {'lam': '-I*sqrt(2*pi)'}), (['fourierOp', '_', 'hermite2', '-Real.sqrt (2 * π)', '_', '_'], False, {'lam': '-sqrt(2*pi)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = SR(eval('lam**4', scope))
    R = SR(eval('(2*pi)**2', scope))
    d = (L - R).simplify_full().canonicalize_radical()
    if bool(d == 0):
        R = L  # proven equal symbolically
    else:
        L, R = num(L), num(R)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
    continue
    L = side(eval('lam**4', scope))
    R = side(eval('(2*pi)**2', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
