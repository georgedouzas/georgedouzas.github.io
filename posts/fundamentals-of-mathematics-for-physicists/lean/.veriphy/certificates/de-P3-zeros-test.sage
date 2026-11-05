# name: de-P3-zeros-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.P3_zeros file "de-P3-zeros-test.sage"
# description: P₃ = 0 iff x = 0 or 5x² = 3, at x = 0 (edge), √(3/5), 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=B((5*(0)^3-3*(0))/2 == 0),
                    rhs=B((0) == 0 or 5*(0)^2 == 3), relation='==', edge=True),
    oracle_instance('sqrt(3/5)',
                    lhs=B((5*(sqrt(3/5))^3-3*(sqrt(3/5)))/2 == 0),
                    rhs=B((sqrt(3/5)) == 0 or 5*(sqrt(3/5))^2 == 3), relation='==', edge=False),
    oracle_instance('1',
                    lhs=B((5*(1)^3-3*(1))/2 == 0),
                    rhs=B((1) == 0 or 5*(1)^2 == 3), relation='==', edge=False),
]
