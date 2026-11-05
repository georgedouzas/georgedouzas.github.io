# name: de-kg-lowest-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Exercises.kg_lowest file "de-kg-lowest-test.sage"
# description: n₁² + n₂² + n₃² ≥ 3 for nᵢ ≥ 1 at (1, 1, 1) (edge), (2, 1, 3), (5, 5, 1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '1', '_', '_', '_',
                    lhs=1^2 + 1^2 + 1^2,
                    rhs=3, relation='>=', edge=True),
    oracle_instance('2', '1', '3', '_', '_', '_',
                    lhs=2^2 + 1^2 + 3^2,
                    rhs=3, relation='>=', edge=False),
    oracle_instance('5', '5', '1', '_', '_', '_',
                    lhs=5^2 + 5^2 + 1^2,
                    rhs=3, relation='>=', edge=False),
]
