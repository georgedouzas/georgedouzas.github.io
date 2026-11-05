# name: de-abs-left-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FundamentalSolutions.abs_left file "de-abs-left-test.sage"
# description: d/dx |x|/2 = -1/2 for x < 0 by Sage at x = -1 (edge), -1/10, -7
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('-1', '_',
                    lhs=diff(abs(x)/2, x).subs(x=-1),
                    rhs=-1/2, relation='==', edge=True),
    oracle_instance('-1/10', '_',
                    lhs=diff(abs(x)/2, x).subs(x=-1/10),
                    rhs=-1/2, relation='==', edge=False),
    oracle_instance('-7', '_',
                    lhs=diff(abs(x)/2, x).subs(x=-7),
                    rhs=-1/2, relation='==', edge=False),
]
