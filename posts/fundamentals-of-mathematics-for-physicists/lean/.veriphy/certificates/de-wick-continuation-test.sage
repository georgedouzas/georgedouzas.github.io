# name: de-wick-continuation-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FeynmanPrescription.wick_continuation file "de-wick-continuation-test.sage"
# description: i e^{-ω(it)}/(2ω) = (i/2ω)e^{-iωt} by Sage at (ω, t) = (1, 0) (edge), (2, 1), (1/2, 3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '0', '_',
                    lhs=I*(exp(-(1)*(I*(0)))/(2*(1))),
                    rhs=I/(2*(1))*exp(-I*(1)*(0)), relation='==', edge=True),
    oracle_instance('2', '1', '_',
                    lhs=I*(exp(-(2)*(I*(1)))/(2*(2))),
                    rhs=I/(2*(2))*exp(-I*(2)*(1)), relation='==', edge=False),
    oracle_instance('1/2', '3', '_',
                    lhs=I*(exp(-(1/2)*(I*(3)))/(2*(1/2))),
                    rhs=I/(2*(1/2))*exp(-I*(1/2)*(3)), relation='==', edge=False),
]
