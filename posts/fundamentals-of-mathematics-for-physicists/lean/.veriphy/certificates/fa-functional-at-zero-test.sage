# name: fa-functional-at-zero-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Distributions.functional_at_zero file "fa-functional-at-zero-test.sage"
# description: for T = c δ (c = 3) and ψ = e^{-x²}, Sage evaluates T φ = T ψ · φ(0) on φ = cos, x + 2, eˣ
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
CASES = [(['3 • δ', '_', 'fun x => Real.exp (-x ^ 2)', '_', 'Real.cos'], True, {'v': 'cos(0)'}), (['3 • δ', '_', 'fun x => Real.exp (-x ^ 2)', '_', 'fun x => x + 2'], False, {'v': 'SR(0) + 2'}), (['3 • δ', '_', 'fun x => Real.exp (-x ^ 2)', '_', 'Real.exp'], False, {'v': 'exp(0)'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = SR(eval('3*v', scope))
    R = SR(eval('3*exp(0)*v', scope))
    d = (L - R).simplify_full().canonicalize_radical()
    if bool(d == 0):
        R = L  # proven equal symbolically
    else:
        L, R = num(L), num(R)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
    continue
    L = side(eval('3*v', scope))
    R = side(eval('3*exp(0)*v', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
