# name: de-sign-preserved-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Classification.sign_preserved file "de-sign-preserved-test.sage"
# description: the sign of (det J)² det M is that of det M, at (det J, det M) = (1, 0) (edge), (-2, 3), (1/2, -5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '0', '_',
                    lhs=sign((1)^2*(0)),
                    rhs=sign(0), relation='==', edge=True),
    oracle_instance('-2', '3', '_',
                    lhs=sign((-2)^2*(3)),
                    rhs=sign(3), relation='==', edge=False),
    oracle_instance('1/2', '-5', '_',
                    lhs=sign((1/2)^2*(-5)),
                    rhs=sign(-5), relation='==', edge=False),
]
