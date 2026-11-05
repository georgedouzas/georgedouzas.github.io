# name: de-sin-orthogonal-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.sin_orthogonal file "de-sin-orthogonal-test.sage"
# description: ∫₀^π sin mx sin nx = 0 by Sage at (m, n) = (0, 1) (edge), (1, 2), (3, 7)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '1', '_',
                    lhs=integrate(sin(0*x)*sin(1*x), x, 0, pi),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1', '2', '_',
                    lhs=integrate(sin(1*x)*sin(2*x), x, 0, pi),
                    rhs=0, relation='==', edge=False),
    oracle_instance('3', '7', '_',
                    lhs=integrate(sin(3*x)*sin(7*x), x, 0, pi),
                    rhs=0, relation='==', edge=False),
]
