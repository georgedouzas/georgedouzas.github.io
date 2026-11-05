# name: de-harmonic-general-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.harmonic_general file "de-harmonic-general-test.sage"
# description: every solution of u'' + u = 0 is u(0) cos + u'(0) sin: Sage's desolve with u(0) = a, u'(0) = b at (a, b) = (0, 0) (edge), (1, 2), (-3, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
U = function('U')
instances = [
    oracle_instance('_', '_', '_', '_', '1',
                    lhs=desolve(diff(U(x),x,2) + U(x) == 0, U(x), ics=[0, 0, 0]),
                    rhs=(0)*cos(x) + (0)*sin(x), relation='==', edge=True),
    oracle_instance('_', '_', '_', '_', '1',
                    lhs=desolve(diff(U(x),x,2) + U(x) == 0, U(x), ics=[0, 1, 2]),
                    rhs=(1)*cos(x) + (2)*sin(x), relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '1',
                    lhs=desolve(diff(U(x),x,2) + U(x) == 0, U(x), ics=[0, -3, 1/2]),
                    rhs=(-3)*cos(x) + (1/2)*sin(x), relation='==', edge=False),
]
