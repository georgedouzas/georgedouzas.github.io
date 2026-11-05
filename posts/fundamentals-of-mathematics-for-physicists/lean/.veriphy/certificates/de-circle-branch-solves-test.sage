# name: de-circle-branch-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.circle_branch_solves file "de-circle-branch-solves-test.sage"
# description: (√(2C - x²))' = -x/√(2C - x²) by Sage at (C, x) = (1, 0) (edge), (2, 1), (5/2, -2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('1', '0', '_',
                    lhs=(diff(sqrt(2*(1) - s_^2), s_) - (-s_/sqrt(2*(1) - s_^2))).subs(s_=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '1', '_',
                    lhs=(diff(sqrt(2*(2) - s_^2), s_) - (-s_/sqrt(2*(2) - s_^2))).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('5/2', '-2', '_',
                    lhs=(diff(sqrt(2*(5/2) - s_^2), s_) - (-s_/sqrt(2*(5/2) - s_^2))).subs(s_=-2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
