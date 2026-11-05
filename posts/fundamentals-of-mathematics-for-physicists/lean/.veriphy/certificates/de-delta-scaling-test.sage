# name: de-delta-scaling-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.delta_scaling file "de-delta-scaling-test.sage"
# description: ∫g(ax)f(x) = |a⁻¹|∫g(y)f(y/a) by Sage for g = e^{-x²}, f = cos and f = 1 + x² at a = 2, -1/2, and a = 1 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('_', '_', '2', '_',
                    lhs=integrate(exp(-(2*x)^2)*cos(x), x, -oo, oo),
                    rhs=1/2*integrate(exp(-x^2)*cos(x/2), x, -oo, oo), relation='==', edge=False),
    oracle_instance('_', '_', '-1/2', '_',
                    lhs=integrate(exp(-(x/2)^2)*(1 + x^2), x, -oo, oo),
                    rhs=2*integrate(exp(-x^2)*(1 + (2*x)^2), x, -oo, oo), relation='==', edge=False),
    oracle_instance('_', '_', '1', '_',
                    lhs=integrate(exp(-x^2)*cos(x), x, -oo, oo),
                    rhs=integrate(exp(-x^2)*cos(x), x, -oo, oo), relation='==', edge=True),
]
