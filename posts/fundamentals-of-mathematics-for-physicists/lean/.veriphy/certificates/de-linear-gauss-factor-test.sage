# name: de-linear-gauss-factor-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.linear_gauss_factor file "de-linear-gauss-factor-test.sage"
# description: (e^{-x²}(-1/2 + Ce^{x²}))' = xe^{-x²} by Sage at (C, x) = (0, 0) (edge), (1, 1), (-3, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('0', '0',
                    lhs=(diff(exp(-s_^2)*(-1/2 + (0)*exp(s_^2)), s_) - (s_*exp(-s_^2))).subs(s_=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1', '1',
                    lhs=(diff(exp(-s_^2)*(-1/2 + (1)*exp(s_^2)), s_) - (s_*exp(-s_^2))).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-3', '1/2',
                    lhs=(diff(exp(-s_^2)*(-1/2 + (-3)*exp(s_^2)), s_) - (s_*exp(-s_^2))).subs(s_=1/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
