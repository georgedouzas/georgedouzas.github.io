# name: de-gradient-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LaplaceEquation.gradient_zero file "de-gradient-zero-test.sage"
# description: w_x² + w_y² = 0 forces w_x = w_y = 0: at (0, 0) (edge), and Sage's real solutions of the equation
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
a_, b_ = var('a_ b_')
instances = [
    oracle_instance('0', '0', '_',
                    lhs=0^2 + 0^2,
                    rhs=0, relation='==', edge=True),
    oracle_instance('0', '0', '_',
                    lhs=B(all(s[a_] == 0 and s[b_] == 0 for s in solve(a_^2 + b_^2 == 0, a_, b_, solution_dict=True) if s[a_].is_real())),
                    rhs=1, relation='==', edge=False),
    oracle_instance('0', '0', '_',
                    lhs=B(not bool(RIF(1/10)^2 + RIF(0)^2 == 0)),
                    rhs=1, relation='==', edge=False),
]
