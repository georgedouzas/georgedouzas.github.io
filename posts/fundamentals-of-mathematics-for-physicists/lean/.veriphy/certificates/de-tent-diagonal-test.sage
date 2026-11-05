# name: de-tent-diagonal-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.GreensBoundary.tent_diagonal file "de-tent-diagonal-test.sage"
# description: G(ξ, ξ) = ξ(1 - ξ) exactly at ξ = 0 (edge), 1/4, 3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=min(0,0)*(1 - max(0,0)),
                    rhs=(0)*(1 - (0)), relation='==', edge=True),
    oracle_instance('1/4',
                    lhs=min(1/4,1/4)*(1 - max(1/4,1/4)),
                    rhs=(1/4)*(1 - (1/4)), relation='==', edge=False),
    oracle_instance('3',
                    lhs=min(3,3)*(1 - max(3,3)),
                    rhs=(3)*(1 - (3)), relation='==', edge=False),
]
