# name: de-cos-sin-orth-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.cos_sin_orthogonal file "de-cos-sin-orth-test.sage"
# description: ∫_{-π}^π cos mx sin nx = 0 by Sage at (m, n) = (0, 0) (edge), (1, 1), (2, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=integrate(cos((0)*x)*sin((0)*x), x, -pi, pi),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1', '1',
                    lhs=integrate(cos((1)*x)*sin((1)*x), x, -pi, pi),
                    rhs=0, relation='==', edge=False),
    oracle_instance('2', '1/2',
                    lhs=integrate(cos((2)*x)*sin((1/2)*x), x, -pi, pi),
                    rhs=0, relation='==', edge=False),
]
