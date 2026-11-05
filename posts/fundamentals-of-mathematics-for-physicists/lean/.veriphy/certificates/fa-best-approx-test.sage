# name: fa-best-approx-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.Parseval.best_approx file "fa-best-approx-test.sage"
# description: in ℝ³ with explicit orthonormal pairs Sage evaluates both sides exactly
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
def ba(E, f, t):
    f = vector(SR, f)
    v = f - sum(ti * vector(SR, ei) for ti, ei in zip(t, E))
    c = [vector(SR, ei).dot_product(f) for ei in E]
    return v.dot_product(v), f.dot_product(f) - sum(ci**2 for ci in c) + sum((ci - ti)**2 for ci, ti in zip(c, t))
instances = []
CASES = [(['EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 2', '_', 'Finset.univ', 'e₁₂', '_', '![1, 2, 3]', '![0, 0]'], True, {'t1': '0', 't2': '0', 'r': '0'}), (['EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 2', '_', 'Finset.univ', 'e₁₂', '_', '![1, 2, 3]', '![1, 5]'], False, {'t1': '1', 't2': '5', 'r': '0'}), (['EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 2', '_', 'Finset.univ', 'rotated', '_', '![3, 1, 2]', '![2, -1]'], False, {'t1': '2', 't2': '-1', 'r': '1'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('ba([[1,0,0],[0,1,0]] if r == 0 else [[1/sqrt(2),1/sqrt(2),0],[1/sqrt(2),-1/sqrt(2),0]], [1,2,3] if r == 0 else [3,1,2], [t1, t2])[0]', scope))
    R = side(eval('ba([[1,0,0],[0,1,0]] if r == 0 else [[1/sqrt(2),1/sqrt(2),0],[1/sqrt(2),-1/sqrt(2),0]], [1,2,3] if r == 0 else [3,1,2], [t1, t2])[1]', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
