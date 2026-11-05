# name: de-nonunique-neg-sq-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.InitialValueProblems.nonunique_neg_sq file "de-nonunique-neg-sq-test.sage"
# description: (-x²)' = 2√|-x²| at x = 0 (edge), -1, -3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '_',
                    lhs=diff(-x^2, x).subs(x=0),
                    rhs=0, relation='==', edge=True),
    oracle_instance('-1', '_',
                    lhs=diff(-x^2, x).subs(x=-1),
                    rhs=2*sqrt(abs(-(-1)^2)), relation='==', edge=False),
    oracle_instance('-3', '_',
                    lhs=diff(-x^2, x).subs(x=-3),
                    rhs=2*sqrt(abs(-(-3)^2)), relation='==', edge=False),
]
