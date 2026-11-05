# name: fa-wigner-exponential-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.MatrixCoefficients.WignerExponential file "fa-wigner-exponential-test.sage"
# description: Sage sums the exponential series of θA to order 90 in 128-bit intervals (with its tail bound) and compares with d¹(θ) entrywise
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
A = matrix(SR, [[0, -sqrt(2)/2, 0], [sqrt(2)/2, 0, -sqrt(2)/2], [0, sqrt(2)/2, 0]])
d1 = lambda t: matrix(SR, [[(1 + cos(t))/2, -(sqrt(2)/2)*sin(t), (1 - cos(t))/2], [sqrt(2)/2*sin(t), cos(t), -(sqrt(2)/2)*sin(t)], [(1 - cos(t))/2, sqrt(2)/2*sin(t), (1 + cos(t))/2]])
def diffnorm(t):
    At = matrix(RI, 3, 3, [RI(SR(t)*x) for x in A.list()])
    E = identity_matrix(RI, 3)
    term = identity_matrix(RI, 3)
    for k in range(1, 90):
        term = term*At/k
        E += term
    nrm = RI(abs(SR(t)))*RI(sqrt(2))
    tail = nrm**90/factorial(90)*nrm.exp()
    D = d1(t)
    return sum(abs(E[i, j] - RI(D[i, j])) for i in range(3) for j in range(3)) + RI(-tail.upper(), tail.upper())
instances = []
CASES = [(['0'], True, {'t': '0'}), (['1'], False, {'t': '1'}), (['5/2'], False, {'t': '5/2'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('diffnorm(t)', scope))
    R = side(eval('0', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
