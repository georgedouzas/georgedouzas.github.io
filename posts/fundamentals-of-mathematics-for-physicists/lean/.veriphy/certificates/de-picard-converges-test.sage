# name: de-picard-converges-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ODEComputations.picard_linear_converges file "de-picard-converges-test.sage"
# description: the Taylor sums of e^{ax} converge: the partial sum of 80 terms is within 10⁻¹⁵ of e^{ax} in interval arithmetic at (a, x) = (0, 5) (edge), (1, 1), (-2, 3/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '5',
                    lhs=RIF(abs(sum((QQ(0)*QQ(5))^j/factorial(j) for j in range(80)) - exp(QQ(0)*QQ(5)))),
                    rhs=RIF(10^-15), relation='<', edge=True),
    oracle_instance('1', '1',
                    lhs=RIF(abs(sum((QQ(1)*QQ(1))^j/factorial(j) for j in range(80)) - exp(QQ(1)*QQ(1)))),
                    rhs=RIF(10^-15), relation='<', edge=False),
    oracle_instance('-2', '3/2',
                    lhs=RIF(abs(sum((QQ(-2)*QQ(3/2))^j/factorial(j) for j in range(80)) - exp(QQ(-2)*QQ(3/2)))),
                    rhs=RIF(10^-15), relation='<', edge=False),
]
