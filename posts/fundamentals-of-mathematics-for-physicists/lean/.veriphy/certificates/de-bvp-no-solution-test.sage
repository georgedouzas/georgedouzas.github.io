# name: de-bvp-no-solution-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.bvp_no_solution file "de-bvp-no-solution-test.sage"
# description: u(0) = 0 forces u(π) = 0: Sage's desolve with u(0) = 0, u'(0) = b, evaluated at π, for b = 0 (edge), 1, -5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
U = function('U')
instances = [
    oracle_instance('_', '_', '_', '_', '_',
                    lhs=desolve(diff(U(x),x,2) + U(x) == 0, U(x), ics=[0, 0, 0]).subs(x=pi),
                    rhs=0, relation='==', edge=True),
    oracle_instance('_', '_', '_', '_', '_',
                    lhs=desolve(diff(U(x),x,2) + U(x) == 0, U(x), ics=[0, 0, 1]).subs(x=pi),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_',
                    lhs=desolve(diff(U(x),x,2) + U(x) == 0, U(x), ics=[0, 0, -5]).subs(x=pi),
                    rhs=0, relation='==', edge=False),
]
