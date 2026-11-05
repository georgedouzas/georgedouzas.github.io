# name: de-stable-complex-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.stable_complex_iff file "de-stable-complex-test.sage"
# description: α < 0 iff 2α < 0 and α² + β² > 0, at (α, β) = (0, 1) (edge), (-1, 2), (3, -1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '1', '_',
                    lhs=B((0) < 0),
                    rhs=B(2*(0) < 0 and (0)^2 + (1)^2 > 0), relation='==', edge=True),
    oracle_instance('-1', '2', '_',
                    lhs=B((-1) < 0),
                    rhs=B(2*(-1) < 0 and (-1)^2 + (2)^2 > 0), relation='==', edge=False),
    oracle_instance('3', '-1', '_',
                    lhs=B((3) < 0),
                    rhs=B(2*(3) < 0 and (3)^2 + (-1)^2 > 0), relation='==', edge=False),
]
