# name: de-logistic-family-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.logistic_family_solves file "de-logistic-family-solves-test.sage"
# description: y = 1/(1 + Ce^{-x}) has y' = y(1 - y) by Sage at (C, x) = (0, 1) (edge), (2, 0), (-1/2, 1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('0', '1', '_',
                    lhs=(diff(1/(1+(0)*exp(-s_)), s_) - (1/(1+(0)*exp(-s_))*(1 - 1/(1+(0)*exp(-s_))))).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '0', '_',
                    lhs=(diff(1/(1+(2)*exp(-s_)), s_) - (1/(1+(2)*exp(-s_))*(1 - 1/(1+(2)*exp(-s_))))).subs(s_=0).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1/2', '1', '_',
                    lhs=(diff(1/(1+(-1/2)*exp(-s_)), s_) - (1/(1+(-1/2)*exp(-s_))*(1 - 1/(1+(-1/2)*exp(-s_))))).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
