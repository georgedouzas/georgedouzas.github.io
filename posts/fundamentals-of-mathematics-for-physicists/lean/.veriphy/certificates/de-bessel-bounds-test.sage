# name: de-bessel-bounds-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_bounds file "de-bessel-bounds-test.sage"
# description: the alternating partial sums bracket Sage's J₀ in interval arithmetic at (x, k) = (0, 0) (edge), (2, 2), (4, 5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
def g(i, x): return (x/2)^(2*i+2)/factorial(i+1)^2
RIF = RealIntervalField(100)
instances = [
    oracle_instance('0', '_', '_', '0',
                    lhs=B(RIF(1 - sum((-1)^i*g(i, QQ(0)) for i in range(2*0+1))) <= RIF(bessel_J(0, 0)) and RIF(bessel_J(0, 0)) <= RIF(1 - sum((-1)^i*g(i, QQ(0)) for i in range(2*0)))),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '_', '_', '2',
                    lhs=B(RIF(1 - sum((-1)^i*g(i, QQ(2)) for i in range(2*2+1))) <= RIF(bessel_J(0, 2)) and RIF(bessel_J(0, 2)) <= RIF(1 - sum((-1)^i*g(i, QQ(2)) for i in range(2*2)))),
                    rhs=1, relation='==', edge=False),
    oracle_instance('4', '_', '_', '5',
                    lhs=B(RIF(1 - sum((-1)^i*g(i, QQ(4)) for i in range(2*5+1))) <= RIF(bessel_J(0, 4)) and RIF(bessel_J(0, 4)) <= RIF(1 - sum((-1)^i*g(i, QQ(4)) for i in range(2*5)))),
                    rhs=1, relation='==', edge=False),
]
