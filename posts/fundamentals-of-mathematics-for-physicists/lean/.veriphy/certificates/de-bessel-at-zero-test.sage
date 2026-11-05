# name: de-bessel-at-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ_at_zero file "de-bessel-at-zero-test.sage"
# description: Jₙ(0) by Sage's bessel_J at n = 0 (edge), 1, 4
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=bessel_J(0, 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1',
                    lhs=bessel_J(1, 0),
                    rhs=0, relation='==', edge=False),
    oracle_instance('4',
                    lhs=bessel_J(4, 0),
                    rhs=0, relation='==', edge=False),
]
