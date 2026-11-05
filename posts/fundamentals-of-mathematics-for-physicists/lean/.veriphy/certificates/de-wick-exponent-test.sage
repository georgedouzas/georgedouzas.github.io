# name: de-wick-exponent-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FeynmanPrescription.wick_exponent file "de-wick-exponent-test.sage"
# description: e^{i(iω)τ} = e^{-ωτ} by Sage at (ω, τ) = (0, 0) (edge), (1, 2), (3, -1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=exp(I*(I*(0))*(0)),
                    rhs=exp(-(0)*(0)), relation='==', edge=True),
    oracle_instance('1', '2',
                    lhs=exp(I*(I*(1))*(2)),
                    rhs=exp(-(1)*(2)), relation='==', edge=False),
    oracle_instance('3', '-1/2',
                    lhs=exp(I*(I*(3))*(-1/2)),
                    rhs=exp(-(3)*(-1/2)), relation='==', edge=False),
]
