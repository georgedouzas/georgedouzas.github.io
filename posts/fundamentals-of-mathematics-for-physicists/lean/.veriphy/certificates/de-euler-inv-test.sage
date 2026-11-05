# name: de-euler-inv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.euler_inv_solves file "de-euler-inv-test.sage"
# description: x²(1/x)'' - 2/x evaluated exactly at x = 1 (edge), -2, 1/3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=(x^2*diff(1/x, x, 2) - 2/x).subs(x=1),
                    rhs=0, relation='==', edge=True),
    oracle_instance('-2', '_',
                    lhs=(x^2*diff(1/x, x, 2) - 2/x).subs(x=-2),
                    rhs=0, relation='==', edge=False),
    oracle_instance('1/3', '_',
                    lhs=(x^2*diff(1/x, x, 2) - 2/x).subs(x=1/3),
                    rhs=0, relation='==', edge=False),
]
