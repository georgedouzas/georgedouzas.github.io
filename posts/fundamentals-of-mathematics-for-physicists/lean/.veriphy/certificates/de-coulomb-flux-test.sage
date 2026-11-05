# name: de-coulomb-flux-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.coulomb_flux file "de-coulomb-flux-test.sage"
# description: the double integral of the flux density over the sphere is 1 by Sage at R = 1 (edge), 2, 1/7
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
th_ = var('th_')
instances = [
    oracle_instance('1', '_',
                    lhs=integrate(integrate(1/(4*pi*(1)^2)*((1)^2*sin(th_)), th_, 0, pi), x, 0, 2*pi),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '_',
                    lhs=integrate(integrate(1/(4*pi*(2)^2)*((2)^2*sin(th_)), th_, 0, pi), x, 0, 2*pi),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/7', '_',
                    lhs=integrate(integrate(1/(4*pi*(1/7)^2)*((1/7)^2*sin(th_)), th_, 0, pi), x, 0, 2*pi),
                    rhs=1, relation='==', edge=False),
]
