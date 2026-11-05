# name: de-logistic-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.logistic_solves file "de-logistic-solves-test.sage"
# description: y' = y(1 - y) for y = 1/(1 + e^{-x}): Sage's derivative at x = 0 (edge), 1, -3/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('fun s => 1 / (1 + Real.exp (-s))', '_', '0',
                    lhs=(diff(1/(1+exp(-s_)), s_) - (1/(1+exp(-s_))*(1 - 1/(1+exp(-s_))))).subs(s_=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('fun s => 1 / (1 + Real.exp (-s))', '_', '1',
                    lhs=(diff(1/(1+exp(-s_)), s_) - (1/(1+exp(-s_))*(1 - 1/(1+exp(-s_))))).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun s => 1 / (1 + Real.exp (-s))', '_', '-3/2',
                    lhs=(diff(1/(1+exp(-s_)), s_) - (1/(1+exp(-s_))*(1 - 1/(1+exp(-s_))))).subs(s_=-3/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
