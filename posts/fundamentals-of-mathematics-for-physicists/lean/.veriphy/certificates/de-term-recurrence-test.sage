# name: de-term-recurrence-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.term_recurrence file "de-term-recurrence-test.sage"
# description: the termwise recurrence of the Bessel series, exactly at (n, k, x) = (0, 0, 1) (edge), (1, 2, 3), (3, 1, -1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
def tm(n, k, x): return (-1)^k/(factorial(k)*factorial(k+n))*(QQ(x)/2)^(2*k+n)
instances = [
    oracle_instance('0', '0', '1', '_',
                    lhs=tm(0, 0+1, 1) + tm(0+2, 0, 1),
                    rhs=2*(0+1)/(1)*tm(0+1, 0+1, 1), relation='==', edge=True),
    oracle_instance('1', '2', '3', '_',
                    lhs=tm(1, 2+1, 3) + tm(1+2, 2, 3),
                    rhs=2*(1+1)/(3)*tm(1+1, 2+1, 3), relation='==', edge=False),
    oracle_instance('3', '1', '-1/2', '_',
                    lhs=tm(3, 1+1, -1/2) + tm(3+2, 1, -1/2),
                    rhs=2*(3+1)/(-1/2)*tm(3+1, 1+1, -1/2), relation='==', edge=False),
]
