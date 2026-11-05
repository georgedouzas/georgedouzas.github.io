# name: de-g0-summable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.g0_summable file "de-g0-summable-test.sage"
# description: Σgᵢ(x) converges: its 60 partial sums stay below e^{x²/4} in interval arithmetic at x = 0 (edge), 2, 10
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=RealIntervalField(100)(sum((QQ(0)/2)^(2*i+2)/factorial(i+1)^2 for i in range(60))),
                    rhs=RealIntervalField(100)(exp(QQ(0)^2/4)), relation='<=', edge=True),
    oracle_instance('2',
                    lhs=RealIntervalField(100)(sum((QQ(2)/2)^(2*i+2)/factorial(i+1)^2 for i in range(60))),
                    rhs=RealIntervalField(100)(exp(QQ(2)^2/4)), relation='<=', edge=False),
    oracle_instance('10',
                    lhs=RealIntervalField(100)(sum((QQ(10)/2)^(2*i+2)/factorial(i+1)^2 for i in range(60))),
                    rhs=RealIntervalField(100)(exp(QQ(10)^2/4)), relation='<=', edge=False),
]
