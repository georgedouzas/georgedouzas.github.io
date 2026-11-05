# name: de-sine-boundary-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.sine_mode_boundary file "de-sine-boundary-test.sage"
# description: sin(n·0) = 0 and sin(nπ) = 0 by Sage at n = 0 (edge), 1, 7
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=B(sin(0*0) == 0 and sin(0*pi) == 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1',
                    lhs=B(sin(1*0) == 0 and sin(1*pi) == 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('7',
                    lhs=B(sin(7*0) == 0 and sin(7*pi) == 0),
                    rhs=1, relation='==', edge=False),
]
