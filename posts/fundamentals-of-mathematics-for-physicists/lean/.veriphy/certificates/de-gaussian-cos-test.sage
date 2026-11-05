# name: de-gaussian-cos-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.gaussian_against_cos file "de-gaussian-cos-test.sage"
# description: ∫g_ε cos x = e^{-ε²/4} by Sage at ε = 1 (edge), 1/2, 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=integrate(exp(-x^2/(1)^2)/(sqrt(pi)*(1))*cos(x), x, -oo, oo).simplify_full(),
                    rhs=exp(-(1)^2/4), relation='==', edge=True),
    oracle_instance('1/2', '_',
                    lhs=integrate(exp(-x^2/(1/2)^2)/(sqrt(pi)*(1/2))*cos(x), x, -oo, oo).simplify_full(),
                    rhs=exp(-(1/2)^2/4), relation='==', edge=False),
    oracle_instance('2', '_',
                    lhs=integrate(exp(-x^2/(2)^2)/(sqrt(pi)*(2))*cos(x), x, -oo, oo).simplify_full(),
                    rhs=exp(-(2)^2/4), relation='==', edge=False),
]
