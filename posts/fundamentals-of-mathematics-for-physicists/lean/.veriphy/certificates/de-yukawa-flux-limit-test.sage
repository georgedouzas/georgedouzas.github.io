# name: de-yukawa-flux-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FundamentalSolutions.yukawa_flux_limit file "de-yukawa-flux-limit-test.sage"
# description: Sage's limit R → 0 of e^{-mR}(1 + mR) is 1 for m = 0 (edge), 1, 5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=limit(exp(-(0)*x)*(1 + (0)*x), x=0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1',
                    lhs=limit(exp(-(1)*x)*(1 + (1)*x), x=0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('5',
                    lhs=limit(exp(-(5)*x)*(1 + (5)*x), x=0),
                    rhs=1, relation='==', edge=False),
]
