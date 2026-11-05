# name: de-bessel-j0-eq-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_eq file "de-bessel-j0-eq-test.sage"
# description: J₀(x) = 1 - Σ(-1)ⁱgᵢ(x) by Sage's series sum and bessel_J at x = 0 (edge), 1, 5/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=RealIntervalField(100)(1 - sum((-1)^i*((0)/2)^(2*i+2)/factorial(i+1)^2 for i in range(60))),
                    rhs=RealIntervalField(100)(bessel_J(0, 0)), relation='==', edge=True),
    oracle_instance('1',
                    lhs=RealIntervalField(100)(1 - sum((-1)^i*((1)/2)^(2*i+2)/factorial(i+1)^2 for i in range(60))),
                    rhs=RealIntervalField(100)(bessel_J(0, 1)), relation='==', edge=False),
    oracle_instance('5/2',
                    lhs=RealIntervalField(100)(1 - sum((-1)^i*((5/2)/2)^(2*i+2)/factorial(i+1)^2 for i in range(60))),
                    rhs=RealIntervalField(100)(bessel_J(0, 5/2)), relation='==', edge=False),
]
