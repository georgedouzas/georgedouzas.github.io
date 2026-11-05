# name: de-step-midpoint-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PDEComputations.step_midpoint file "de-step-midpoint-test.sage"
# description: ∫₀^∞ K(t, -y) dy = 1/2 by Sage at (κ, t) = (1, 1) (edge), (2, 1/2), (1/5, 3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
y_ = var('y_')
instances = [
    oracle_instance('1', '1', '_', '_',
                    lhs=integrate(exp(-y_^2/(4*(1)*(1)))/sqrt(4*pi*(1)*(1)), y_, 0, oo),
                    rhs=1/2, relation='==', edge=True),
    oracle_instance('2', '1/2', '_', '_',
                    lhs=integrate(exp(-y_^2/(4*(2)*(1/2)))/sqrt(4*pi*(2)*(1/2)), y_, 0, oo),
                    rhs=1/2, relation='==', edge=False),
    oracle_instance('1/5', '3', '_', '_',
                    lhs=integrate(exp(-y_^2/(4*(1/5)*(3)))/sqrt(4*pi*(1/5)*(3)), y_, 0, oo),
                    rhs=1/2, relation='==', edge=False),
]
