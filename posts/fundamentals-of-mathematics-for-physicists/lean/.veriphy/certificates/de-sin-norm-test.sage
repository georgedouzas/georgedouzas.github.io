# name: de-sin-norm-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.sin_norm file "de-sin-norm-test.sage"
# description: ∫₀^π sin² nx = π/2 by Sage at n = 1 (edge), 2, 9
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=integrate(sin(1*x)^2, x, 0, pi),
                    rhs=pi/2, relation='==', edge=True),
    oracle_instance('2', '_',
                    lhs=integrate(sin(2*x)^2, x, 0, pi),
                    rhs=pi/2, relation='==', edge=False),
    oracle_instance('9', '_',
                    lhs=integrate(sin(9*x)^2, x, 0, pi),
                    rhs=pi/2, relation='==', edge=False),
]
