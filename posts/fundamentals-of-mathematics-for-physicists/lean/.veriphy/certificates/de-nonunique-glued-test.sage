# name: de-nonunique-glued-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.InitialValueProblems.nonunique_glued file "de-nonunique-glued-test.sage"
# description: the glued solution's one-sided difference quotients and 2√|y| at (a, x) = (0, 0) (edge, kink), (1, 3), (2, 1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
h = var('h')
instances = [
    oracle_instance('0', '0',
                    lhs=B(limit(max_symbolic(h, 0)^2/h, h=0, dir='+') == 0 and limit(max_symbolic(h,0)^2/h, h=0, dir='-') == 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1', '3',
                    lhs=diff((x-1)^2, x).subs(x=3),
                    rhs=2*sqrt(abs((3-1)^2)), relation='==', edge=False),
    oracle_instance('2', '1',
                    lhs=0,
                    rhs=2*sqrt(abs(max_symbolic(1-2, 0)^2)), relation='==', edge=False),
]
