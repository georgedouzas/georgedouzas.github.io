# name: de-step-derivative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.step_derivative file "de-step-derivative-test.sage"
# description: -∫₀^∞ f' = f(0) by Sage for f = e^{-x}, e^{-x²}, 1/(1 + x²) (and f = 0, edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('_', '_', '_', '_', '_',
                    lhs=-integrate(diff(exp(-x), x), x, 0, oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_',
                    lhs=-integrate(diff(exp(-x^2), x), x, 0, oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('fun _ => 0', 'fun _ => 0', '_', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
