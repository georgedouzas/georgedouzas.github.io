# name: de-sage-complex-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearSystemComputations.sage_complex_entry file "de-sage-complex-test.sage"
# description: ½e^{(-1+i)t} + ½e^{(-1-i)t} = e^{-t} cos t by Sage's rectform at t = 0 (edge), 1, π
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=SR(1/2*exp((-1+I)*(0)) + 1/2*exp((-1-I)*(0))).rectform().simplify_full(),
                    rhs=exp(-(0))*cos(0), relation='==', edge=True),
    oracle_instance('1',
                    lhs=SR(1/2*exp((-1+I)*(1)) + 1/2*exp((-1-I)*(1))).rectform().simplify_full(),
                    rhs=exp(-(1))*cos(1), relation='==', edge=False),
    oracle_instance('pi',
                    lhs=SR(1/2*exp((-1+I)*(pi)) + 1/2*exp((-1-I)*(pi))).rectform().simplify_full(),
                    rhs=exp(-(pi))*cos(pi), relation='==', edge=False),
]
