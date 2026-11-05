# name: de-tent-symm-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.GreensBoundary.tent_symm file "de-tent-symm-test.sage"
# description: x_<(1 - x_>) is symmetric, exactly at (0, 0) (edge), (1/3, 3/4), (2, -1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=min(0,0)*(1 - max(0,0)),
                    rhs=min(0,0)*(1 - max(0,0)), relation='==', edge=True),
    oracle_instance('1/3', '3/4',
                    lhs=min(1/3,3/4)*(1 - max(1/3,3/4)),
                    rhs=min(3/4,1/3)*(1 - max(3/4,1/3)), relation='==', edge=False),
    oracle_instance('2', '-1',
                    lhs=min(2,-1)*(1 - max(2,-1)),
                    rhs=min(-1,2)*(1 - max(-1,2)), relation='==', edge=False),
]
