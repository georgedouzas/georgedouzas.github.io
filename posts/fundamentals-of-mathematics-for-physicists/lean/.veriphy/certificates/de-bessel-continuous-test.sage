# name: de-bessel-continuous-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_continuousOn file "de-bessel-continuous-test.sage"
# description: J₀ is continuous on [-R, R]: Sage's bessel_J has J₀(x) → J₀(x₀) (difference below 10⁻⁶ at x₀ ± 10⁻⁸, interval arithmetic) at x₀ = 0 (edge), R/2, R for R = 4
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('4',
                    lhs=RealIntervalField(100)(abs(bessel_J(0, 0 + 1/10^8) - bessel_J(0, 0))),
                    rhs=RealIntervalField(100)(1/10^6), relation='<', edge=True),
    oracle_instance('4',
                    lhs=RealIntervalField(100)(abs(bessel_J(0, 2 + 1/10^8) - bessel_J(0, 2))),
                    rhs=RealIntervalField(100)(1/10^6), relation='<', edge=False),
    oracle_instance('4',
                    lhs=RealIntervalField(100)(abs(bessel_J(0, 4 + 1/10^8) - bessel_J(0, 4))),
                    rhs=RealIntervalField(100)(1/10^6), relation='<', edge=False),
]
