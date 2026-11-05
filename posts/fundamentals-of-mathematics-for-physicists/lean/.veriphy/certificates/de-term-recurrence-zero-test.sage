# name: de-term-recurrence-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.term_recurrence_zero file "de-term-recurrence-zero-test.sage"
# description: the first terms: term n 0 = (2(n+1)/x) term (n+1) 0, exactly at (n, x) = (0, 1) (edge), (2, 3), (5, -2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
def tm(n, k, x): return (-1)^k/(factorial(k)*factorial(k+n))*(QQ(x)/2)^(2*k+n)
instances = [
    oracle_instance('0', '1', '_',
                    lhs=tm(0, 0, 1),
                    rhs=2*(0+1)/(1)*tm(0+1, 0, 1), relation='==', edge=True),
    oracle_instance('2', '3', '_',
                    lhs=tm(2, 0, 3),
                    rhs=2*(2+1)/(3)*tm(2+1, 0, 3), relation='==', edge=False),
    oracle_instance('5', '-2', '_',
                    lhs=tm(5, 0, -2),
                    rhs=2*(5+1)/(-2)*tm(5+1, 0, -2), relation='==', edge=False),
]
