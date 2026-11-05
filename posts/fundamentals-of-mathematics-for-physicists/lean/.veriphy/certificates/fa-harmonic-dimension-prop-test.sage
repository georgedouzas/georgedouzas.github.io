# name: fa-harmonic-dimension-prop-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates in interval arithmetic
# goal: sage_test KaluzaKlein.FourierAnalysis.HarmonicComputations.HarmonicDimension file "fa-harmonic-dimension-prop-test.sage"
# description: Sage builds the matrix of ∇² from the monomials of degree l to those of degree l - 2 over ℚ, computes its rank C(l, 2) and the kernel dimension 2l + 1
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
Rx = PolynomialRing(QQ, 'x,y,z')
def kerdim(l):
    l = int(l)
    mons = [m for m in Rx.monomials_of_degree(l)]
    if l < 2:
        return len(mons)
    low = Rx.monomials_of_degree(l - 2)
    M = matrix(QQ, [[(sum(m.derivative(v, 2) for v in Rx.gens())).monomial_coefficient(mo) for m in mons] for mo in low])
    return len(mons) - M.rank()
instances = []
CASES = [(['0', '_'], True, {'l': '0'}), (['4', '_'], False, {'l': '4'}), (['5', '_'], False, {'l': '5'}), (['6', '_'], False, {'l': '6'})]
for values, edge, env in CASES:
    env = {k: (eval(v, globals()) if isinstance(v, str) and v.startswith('lambda') else SR(v)) for k, v in env.items()}
    scope = dict(globals()); scope.update(env)
    L = side(eval('kerdim(l)', scope))
    R = side(eval('2*l + 1', scope))
    if L.parent() is CI or R.parent() is CI:
        d = CI(L) - CI(R)
        L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
