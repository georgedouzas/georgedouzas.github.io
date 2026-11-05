# name: de-positive-case-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.positive_case_integer file "de-positive-case-test.sage"
# description: sin kπ = 0 iff k is a positive integer, at k = 1 (edge), 5/2, 3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=B(sin((1)*pi) == 0),
                    rhs=B(QQ(1) in ZZ and QQ(1) >= 1), relation='==', edge=True),
    oracle_instance('5/2', '_',
                    lhs=B(sin((5/2)*pi) == 0),
                    rhs=B(QQ(5/2) in ZZ and QQ(5/2) >= 1), relation='==', edge=False),
    oracle_instance('3', '_',
                    lhs=B(sin((3)*pi) == 0),
                    rhs=B(QQ(3) in ZZ and QQ(3) >= 1), relation='==', edge=False),
]
