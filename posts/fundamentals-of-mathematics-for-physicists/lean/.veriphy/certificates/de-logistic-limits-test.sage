# name: de-logistic-limits-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.logistic_limits file "de-logistic-limits-test.sage"
# description: both limits of 1/(1 + C e^{-x}) computed by Sage's limit at C = 1 (edge), 2, 1/3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=B(limit(1/(1+1*exp(-x)), x=oo) == 1 and limit(1/(1+1*exp(-x)), x=-oo) == 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '_',
                    lhs=B(limit(1/(1+2*exp(-x)), x=oo) == 1 and limit(1/(1+2*exp(-x)), x=-oo) == 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/3', '_',
                    lhs=B(limit(1/(1+exp(-x)/3), x=oo) == 1 and limit(1/(1+exp(-x)/3), x=-oo) == 0),
                    rhs=1, relation='==', edge=False),
]
