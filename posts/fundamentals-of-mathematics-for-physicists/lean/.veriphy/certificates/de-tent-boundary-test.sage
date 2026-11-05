# name: de-tent-boundary-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.GreensBoundary.tent_boundary file "de-tent-boundary-test.sage"
# description: G(0, ξ) = G(1, ξ) = 0 exactly at ξ = 0 (edge), 1/2, 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '_', '_',
                    lhs=B(min(0, 0)*(1 - max(0, 0)) == 0 and min(1, 0)*(1 - max(1, 0)) == 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1/2', '_', '_',
                    lhs=B(min(0, 1/2)*(1 - max(0, 1/2)) == 0 and min(1, 1/2)*(1 - max(1, 1/2)) == 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1', '_', '_',
                    lhs=B(min(0, 1)*(1 - max(0, 1)) == 0 and min(1, 1)*(1 - max(1, 1)) == 0),
                    rhs=1, relation='==', edge=False),
]
