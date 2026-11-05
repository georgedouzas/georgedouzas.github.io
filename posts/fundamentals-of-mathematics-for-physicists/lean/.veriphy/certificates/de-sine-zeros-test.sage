# name: de-sine-zeros-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.sine_zeros file "de-sine-zeros-test.sage"
# description: the zeros of sin nx in (0, π) are kπ/n: Sage's solve for n = 1 (edge, none), 3, 4 counted
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_', 'Real.pi / 2', '_',
                    lhs=len([k for k in range(1, 1) ]),
                    rhs=0, relation='==', edge=True),
    oracle_instance('3', '_', 'Real.pi / 3', '_',
                    lhs=B(sin(3*(pi/3)) == 0 and sin(3*(pi/2)) != 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('4', '_', '3 * Real.pi / 4', '_',
                    lhs=len([k for k in range(1, 4) if sin(4*(k*pi/4)) == 0]),
                    rhs=3, relation='==', edge=False),
]
