# name: de-bessel-bracket-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.besselJ0_zero_of_bracket file "de-bessel-bracket-test.sage"
# description: a sign change of the bounds brackets a zero: Sage's find_root of bessel_J(0, x) lies in (a, b) for (a, b) = (2.404825, 2.404826), (2.4, 2.41) (edge: the textbook bracket), (2.40482, 2.40483)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
RIF = RealIntervalField(53)
instances = [
    oracle_instance('2.404825', '2.404826', '_', '_', '_', '_', '_',
                    lhs=RIF(find_root(bessel_J(0, x), 2, 3)),
                    rhs=RIF(2.404825, 2.404826), relation='==', edge=False),
    oracle_instance('2.4', '2.41', '_', '_', '_', '_', '_',
                    lhs=RIF(find_root(bessel_J(0, x), 2, 3)),
                    rhs=RIF(2.4, 2.41), relation='==', edge=True),
    oracle_instance('2.40482', '2.40483', '_', '_', '_', '_', '_',
                    lhs=RIF(find_root(bessel_J(0, x), 2, 3)),
                    rhs=RIF(2.40482, 2.40483), relation='==', edge=False),
]
