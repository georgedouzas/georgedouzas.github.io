# name: de-real-roots-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Eigenvalues.real_roots_iff file "de-real-roots-test.sage"
# description: λ² - Tλ + D has a real root iff 4D ≤ T²: Sage's real roots at (T, D) = (0, 0) (edge), (2, 2) (none), (3, 2) (two)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=B(len((x^2 - (0)*x + (0)).roots(ring=RR)) > 0),
                    rhs=B(4*(0) <= (0)^2), relation='==', edge=True),
    oracle_instance('2', '2',
                    lhs=B(len((x^2 - (2)*x + (2)).roots(ring=RR)) > 0),
                    rhs=B(4*(2) <= (2)^2), relation='==', edge=False),
    oracle_instance('3', '2',
                    lhs=B(len((x^2 - (3)*x + (2)).roots(ring=RR)) > 0),
                    rhs=B(4*(2) <= (3)^2), relation='==', edge=False),
]
