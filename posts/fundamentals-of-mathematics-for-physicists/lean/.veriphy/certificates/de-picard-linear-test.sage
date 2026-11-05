# name: de-picard-linear-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ODEComputations.picard_linear_iterate file "de-picard-linear-test.sage"
# description: the Picard step of y' = ay maps the degree-k Taylor polynomial to degree k + 1: (a, x, k) = (0, 1, 0) (edge), (2, 1/2, 3), (-1, 3, 5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('0', '1', '0',
                    lhs=1 + integrate((0)*sum(((0)*s_)^j/factorial(j) for j in range(0+1)), s_, 0, 1),
                    rhs=sum(((0)*(1))^j/factorial(j) for j in range(0+2)), relation='==', edge=True),
    oracle_instance('2', '1/2', '3',
                    lhs=1 + integrate((2)*sum(((2)*s_)^j/factorial(j) for j in range(3+1)), s_, 0, 1/2),
                    rhs=sum(((2)*(1/2))^j/factorial(j) for j in range(3+2)), relation='==', edge=False),
    oracle_instance('-1', '3', '5',
                    lhs=1 + integrate((-1)*sum(((-1)*s_)^j/factorial(j) for j in range(5+1)), s_, 0, 3),
                    rhs=sum(((-1)*(3))^j/factorial(j) for j in range(5+2)), relation='==', edge=False),
]
