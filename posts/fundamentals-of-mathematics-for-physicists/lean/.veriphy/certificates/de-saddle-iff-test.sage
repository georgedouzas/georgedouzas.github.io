# name: de-saddle-iff-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.saddle_iff file "de-saddle-iff-test.sage"
# description: a negative product means opposite signs, at (l₁, l₂) = (-1, 1) (edge), (3, -2), (-1/2, 4)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('-1', '1', '_',
                    lhs=B(((-1) < 0 and (1) > 0) or ((-1) > 0 and (1) < 0)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('3', '-2', '_',
                    lhs=B(((3) < 0 and (-2) > 0) or ((3) > 0 and (-2) < 0)),
                    rhs=1, relation='==', edge=False),
    oracle_instance('-1/2', '4', '_',
                    lhs=B(((-1/2) < 0 and (4) > 0) or ((-1/2) > 0 and (4) < 0)),
                    rhs=1, relation='==', edge=False),
]
