# name: de-integral-cos-int-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.integral_cos_int file "de-integral-cos-int-test.sage"
# description: ∫₀^π cos kx = 0 by Sage at k = 1 (edge), -3, 10
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=integrate(cos((1)*x), x, 0, pi),
                    rhs=0, relation='==', edge=True),
    oracle_instance('-3', '_',
                    lhs=integrate(cos((-3)*x), x, 0, pi),
                    rhs=0, relation='==', edge=False),
    oracle_instance('10', '_',
                    lhs=integrate(cos((10)*x), x, 0, pi),
                    rhs=0, relation='==', edge=False),
]
