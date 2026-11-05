# name: de-retarded-outside-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.RetardedGreens.retarded_1d_outside file "de-retarded-outside-test.sage"
# description: ½∫_{x-t}^{x+t}ψ = 0 outside the cone: the interval misses the support, by Sage at (ε, t, x) = (1/2, 1, 2), (1, 0, 3) (edge, t = 0), (1/10, 2, -5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('_', '1/2', '1', '2', '_', '_', '_', '_',
                    lhs=B((2) - (1) >= (1/2) or (2) + (1) <= -(1/2)),
                    rhs=1, relation='==', edge=False),
    oracle_instance('_', '1', '0', '3', '_', '_', '_', '_',
                    lhs=B((3) - (0) >= (1) or (3) + (0) <= -(1)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('_', '1/10', '2', '-5', '_', '_', '_', '_',
                    lhs=B((-5) - (2) >= (1/10) or (-5) + (2) <= -(1/10)),
                    rhs=1, relation='==', edge=False),
]
