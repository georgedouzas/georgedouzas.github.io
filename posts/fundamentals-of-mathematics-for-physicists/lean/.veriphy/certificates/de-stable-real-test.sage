# name: de-stable-real-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.stable_real_iff file "de-stable-real-test.sage"
# description: both real roots negative iff sum < 0 and product > 0, at (l₁, l₂) = (0, 0) (edge), (-1, -3), (2, -1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=B((0) < 0 and (0) < 0),
                    rhs=B((0)+(0) < 0 and (0)*(0) > 0), relation='==', edge=True),
    oracle_instance('-1', '-3',
                    lhs=B((-1) < 0 and (-3) < 0),
                    rhs=B((-1)+(-3) < 0 and (-1)*(-3) > 0), relation='==', edge=False),
    oracle_instance('2', '-1',
                    lhs=B((2) < 0 and (-1) < 0),
                    rhs=B((2)+(-1) < 0 and (2)*(-1) > 0), relation='==', edge=False),
]
