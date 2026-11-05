# name: de-gaussian-unit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.gaussian_unit_integral file "de-gaussian-unit-test.sage"
# description: ∫g_ε = 1 by Sage at ε = 1 (edge), 1/10, 3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=integrate(exp(-x^2/(1)^2)/(sqrt(pi)*(1)), x, -oo, oo),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1/10', '_',
                    lhs=integrate(exp(-x^2/(1/10)^2)/(sqrt(pi)*(1/10)), x, -oo, oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('3', '_',
                    lhs=integrate(exp(-x^2/(3)^2)/(sqrt(pi)*(3)), x, -oo, oo),
                    rhs=1, relation='==', edge=False),
]
