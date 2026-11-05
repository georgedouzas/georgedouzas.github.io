# name: de-P2-zeros-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.P2_zeros file "de-P2-zeros-test.sage"
# description: P₂ = 0 iff 3x² = 1, at x = 0 (edge), 1/√3, 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=B((3*(0)^2-1)/2 == 0),
                    rhs=B(3*(0)^2 == 1), relation='==', edge=True),
    oracle_instance('1/sqrt(3)',
                    lhs=B((3*(1/sqrt(3))^2-1)/2 == 0),
                    rhs=B(3*(1/sqrt(3))^2 == 1), relation='==', edge=False),
    oracle_instance('2',
                    lhs=B((3*(2)^2-1)/2 == 0),
                    rhs=B(3*(2)^2 == 1), relation='==', edge=False),
]
