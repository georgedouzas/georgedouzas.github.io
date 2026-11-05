# name: de-bessel-summable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ_summable file "de-bessel-summable-test.sage"
# description: the series of Jₙ is dominated by (|x|/2)ⁿe^{x²/4}: 60 partial sums of |terms| in interval arithmetic at (n, x) = (0, 0) (edge), (1, 3), (2, -10)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=RealIntervalField(100)(sum(abs((-1)^k/(factorial(k)*factorial(k+0))*(QQ(0)/2)^(2*k+0)) for k in range(60))),
                    rhs=RealIntervalField(100)(abs(QQ(0)/2)^0*exp(QQ(0)^2/4)), relation='<=', edge=True),
    oracle_instance('1', '3',
                    lhs=RealIntervalField(100)(sum(abs((-1)^k/(factorial(k)*factorial(k+1))*(QQ(3)/2)^(2*k+1)) for k in range(60))),
                    rhs=RealIntervalField(100)(abs(QQ(3)/2)^1*exp(QQ(3)^2/4)), relation='<=', edge=False),
    oracle_instance('2', '-10',
                    lhs=RealIntervalField(100)(sum(abs((-1)^k/(factorial(k)*factorial(k+2))*(QQ(-10)/2)^(2*k+2)) for k in range(60))),
                    rhs=RealIntervalField(100)(abs(QQ(-10)/2)^2*exp(QQ(-10)^2/4)), relation='<=', edge=False),
]
