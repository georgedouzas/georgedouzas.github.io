# name: de-cooling-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.cooling_limit file "de-cooling-limit-test.sage"
# description: Sage's limit of T_a + (T₀ - T_a)e^{-kt} at ∞: (k, T_a, T₀) = (1, 0, 0) (edge), (2, 20, 90), (1/2, -3, 5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '0', '0', '_',
                    lhs=limit(0 + 0*exp(-x), x=oo),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '20', '90', '_',
                    lhs=limit(20 + 70*exp(-2*x), x=oo),
                    rhs=20, relation='==', edge=False),
    oracle_instance('1/2', '-3', '5', '_',
                    lhs=limit(-3 + 8*exp(-x/2), x=oo),
                    rhs=-3, relation='==', edge=False),
]
