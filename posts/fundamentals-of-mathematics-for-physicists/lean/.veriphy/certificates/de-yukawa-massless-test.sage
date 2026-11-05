# name: de-yukawa-massless-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FundamentalSolutions.yukawa_massless file "de-yukawa-massless-test.sage"
# description: Sage's limit m → 0 of e^{-mr}/(4πr) is 1/(4πr) for r = 1 (edge), 2, 1/3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1',
                    lhs=limit(exp(-x*(1))/(4*pi*(1)), x=0),
                    rhs=1/(4*pi*(1)), relation='==', edge=True),
    oracle_instance('2',
                    lhs=limit(exp(-x*(2))/(4*pi*(2)), x=0),
                    rhs=1/(4*pi*(2)), relation='==', edge=False),
    oracle_instance('1/3',
                    lhs=limit(exp(-x*(1/3))/(4*pi*(1/3)), x=0),
                    rhs=1/(4*pi*(1/3)), relation='==', edge=False),
]
