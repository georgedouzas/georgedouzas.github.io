# name: de-bessel-recurrence-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ_recurrence file "de-bessel-recurrence-test.sage"
# description: Jₙ + Jₙ₊₂ = (2(n+1)/x)Jₙ₊₁ with Sage's bessel_J at 200 bits (residual below 10⁻⁴⁰) at x = 0.7, 1.7, 5.3 (exr-plb-bessel-recurrence's points) for n + 1 = 1 (edge), 2, 3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '7/10', '_',
                    lhs=RIF(abs((bessel_J(0, 7/10) + bessel_J(2, 7/10) - 2*1/(7/10)*bessel_J(1, 7/10)).n(prec=200))),
                    rhs=RIF(10^-40), relation='<', edge=True),
    oracle_instance('1', '17/10', '_',
                    lhs=RIF(abs((bessel_J(1, 17/10) + bessel_J(3, 17/10) - 2*2/(17/10)*bessel_J(2, 17/10)).n(prec=200))),
                    rhs=RIF(10^-40), relation='<', edge=False),
    oracle_instance('2', '53/10', '_',
                    lhs=RIF(abs((bessel_J(2, 53/10) + bessel_J(4, 53/10) - 2*3/(53/10)*bessel_J(3, 53/10)).n(prec=200))),
                    rhs=RIF(10^-40), relation='<', edge=False),
]
