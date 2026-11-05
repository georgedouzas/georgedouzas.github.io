# name: de-neumann-boundary-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.neumann_boundary file "de-neumann-boundary-test.sage"
# description: (cos nx)' vanishes at 0 and π by Sage, n = 0 (edge), 1, 4
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=B(diff(cos(0*x), x).subs(x=0) == 0 and diff(cos(0*x), x).subs(x=pi) == 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1',
                    lhs=B(diff(cos(1*x), x).subs(x=0) == 0 and diff(cos(1*x), x).subs(x=pi) == 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('4',
                    lhs=B(diff(cos(4*x), x).subs(x=0) == 0 and diff(cos(4*x), x).subs(x=pi) == 0),
                    rhs=1, relation='==', edge=False),
]
