# name: de-kernel-integral-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HeatEquation.kernel_integral file "de-kernel-integral-test.sage"
# description: ∫K dx = 1 by Sage's integral with κ, t > 0 at (κ, t) = (1, 1) (edge), (2, 3), (1/4, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '_', '_',
                    lhs=integrate(exp(-x^2/(4*(1)*(1)))/sqrt(4*pi*(1)*(1)), x, -oo, oo),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '3', '_', '_',
                    lhs=integrate(exp(-x^2/(4*(2)*(3)))/sqrt(4*pi*(2)*(3)), x, -oo, oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/4', '1/2', '_', '_',
                    lhs=integrate(exp(-x^2/(4*(1/4)*(1/2)))/sqrt(4*pi*(1/4)*(1/2)), x, -oo, oo),
                    rhs=1, relation='==', edge=False),
]
