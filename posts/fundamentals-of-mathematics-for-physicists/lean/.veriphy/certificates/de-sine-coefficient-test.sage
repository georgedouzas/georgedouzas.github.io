# name: de-sine-coefficient-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.sine_coefficient file "de-sine-coefficient-test.sage"
# description: the coefficient formula recovers bₙ from a finite sine sum by Sage's integrals: (N, n, b) = (1, 1, 1) (edge), (3, 2, (1, -2, 5)), (4, 4, (0, 0, 0, 7))
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', 'fun _ => 1', '_',
                    lhs=2/pi*integrate(sin(x)*sin(x), x, 0, pi),
                    rhs=1, relation='==', edge=True),
    oracle_instance('3', '2', '_', '_',
                    lhs=2/pi*integrate((sin(x) - 2*sin(2*x) + 5*sin(3*x))*sin(2*x), x, 0, pi),
                    rhs=-2, relation='==', edge=False),
    oracle_instance('4', '4', '_', '_',
                    lhs=2/pi*integrate(7*sin(4*x)*sin(4*x), x, 0, pi),
                    rhs=7, relation='==', edge=False),
]
