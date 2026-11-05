# name: de-sl-nonnegative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.sl_nonnegative file "de-sl-nonnegative-test.sage"
# description: λW = P + Q with W > 0, P, Q ≥ 0 gives λ ≥ 0, at (W, P, Q) = (1, 0, 0) (edge), (2, 3, 1), (1/2, 0, 5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('(0+0)/1', '1', '0', '0', '_', '_', '_', '_',
                    lhs=((0)+(0))/(1),
                    rhs=0, relation='>=', edge=True),
    oracle_instance('(3+1)/2', '2', '3', '1', '_', '_', '_', '_',
                    lhs=((3)+(1))/(2),
                    rhs=0, relation='>=', edge=False),
    oracle_instance('(0+5)/1/2', '1/2', '0', '5', '_', '_', '_', '_',
                    lhs=((0)+(5))/(1/2),
                    rhs=0, relation='>=', edge=False),
]
