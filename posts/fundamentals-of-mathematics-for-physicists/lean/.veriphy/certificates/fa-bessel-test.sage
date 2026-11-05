# name: fa-bessel-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Parseval.bessel file "fa-bessel-test.sage"
# description: in ℝ³ Sage evaluates Σ ⟪e_i, f⟫² ≤ ‖f‖² exactly; equality when f lies in the span
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
def bs(E, f):
    f = vector(SR, f)
    return sum(vector(SR, ei).dot_product(f)**2 for ei in E), f.dot_product(f)
instances = []
CASES = [(['EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 2', '_', 'Finset.univ', 'e₁₂', '_', '![1, 2, 0]'], True, {'r': '0'}), (['EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 2', '_', 'Finset.univ', 'e₁₂', '_', '![1, 2, 3]'], False, {'r': '1'}), (['EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 2', '_', 'Finset.univ', 'rotated', '_', '![3, 1, 2]'], False, {'r': '2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('bs([[1,0,0],[0,1,0]] if r < 2 else [[1/sqrt(2),1/sqrt(2),0],[1/sqrt(2),-1/sqrt(2),0]], [1,2,0] if r == 0 else ([1,2,3] if r == 1 else [3,1,2]))[0]', scope))
    R = side(eval('bs([[1,0,0],[0,1,0]] if r < 2 else [[1/sqrt(2),1/sqrt(2),0],[1/sqrt(2),-1/sqrt(2),0]], [1,2,0] if r == 0 else ([1,2,3] if r == 1 else [3,1,2]))[1]', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='<=', edge=edge))
    print(values, L, R)
