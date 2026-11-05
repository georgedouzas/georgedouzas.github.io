# name: de-logistic-infinity-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ODEComputations.logistic_family_infinity file "de-logistic-infinity-test.sage"
# description: Sage's limit C → ∞ of 1/(1 + Ce^{-x}) at x = 0 (edge), 1, -3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
C_ = var('C_')
instances = [
    oracle_instance('0',
                    lhs=limit(1/(1 + C_*exp(-(0))), C_=oo),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1',
                    lhs=limit(1/(1 + C_*exp(-(1))), C_=oo),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-3',
                    lhs=limit(1/(1 + C_*exp(-(-3))), C_=oo),
                    rhs=0, relation='==', edge=False),
]
