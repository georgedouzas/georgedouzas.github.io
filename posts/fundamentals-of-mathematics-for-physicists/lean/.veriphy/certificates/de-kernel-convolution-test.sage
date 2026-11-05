# name: de-kernel-convolution-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HeatEquation.kernel_convolution file "de-kernel-convolution-test.sage"
# description: the kernel convolved with e^{-y²} is e^{-x²/(1+4κt)}/√(1+4κt), by Sage's integral at (κ, t, x) = (1, 1, 0) (edge), (1/4, 1, 1), (2, 1/2, -3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
y_ = var('y_')
instances = [
    oracle_instance('1', '1', '0', '_', '_',
                    lhs=integrate(exp(-((0) - y_)^2/(4*(1)*(1)))/sqrt(4*pi*(1)*(1))*exp(-y_^2), y_, -oo, oo).simplify_full(),
                    rhs=(exp(-(0)^2/(1+4*(1)*(1)))/sqrt(1+4*(1)*(1))).simplify_full(), relation='==', edge=True),
    oracle_instance('1/4', '1', '1', '_', '_',
                    lhs=integrate(exp(-((1) - y_)^2/(4*(1/4)*(1)))/sqrt(4*pi*(1/4)*(1))*exp(-y_^2), y_, -oo, oo).simplify_full(),
                    rhs=(exp(-(1)^2/(1+4*(1/4)*(1)))/sqrt(1+4*(1/4)*(1))).simplify_full(), relation='==', edge=False),
    oracle_instance('2', '1/2', '-3', '_', '_',
                    lhs=integrate(exp(-((-3) - y_)^2/(4*(2)*(1/2)))/sqrt(4*pi*(2)*(1/2))*exp(-y_^2), y_, -oo, oo).simplify_full(),
                    rhs=(exp(-(-3)^2/(1+4*(2)*(1/2)))/sqrt(1+4*(2)*(1/2))).simplify_full(), relation='==', edge=False),
]
