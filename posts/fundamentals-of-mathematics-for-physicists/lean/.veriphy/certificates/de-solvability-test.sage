# name: de-solvability-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.solvability_condition file "de-solvability-test.sage"
# description: ∫₀^π (u'' + n²u) sin nx = 0 for u vanishing at 0, π: u = x(π - x) with n = 1, 2; u = sin 3x with n = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate((diff(x*(pi-x),x,2) + x*(pi-x))*sin(x), x, 0, pi),
                    rhs=0, relation='==', edge=False),
    oracle_instance('2', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate((diff(x*(pi-x),x,2) + 4*x*(pi-x))*sin(2*x), x, 0, pi),
                    rhs=0, relation='==', edge=False),
    oracle_instance('0', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate((diff(sin(3*x),x,2))*sin(0*x), x, 0, pi),
                    rhs=0, relation='==', edge=True),
]
