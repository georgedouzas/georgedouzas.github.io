# name: de-linear-difference-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.linear_difference file "de-linear-difference-test.sage"
# description: the difference of two solutions of y' + y = e^{-2x} (namely e^{-x} - e^{-2x} and 2e^{-x} - e^{-2x}) solves y' + y = 0, at x = 0 (edge), 1, -2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun _ => 1', 'fun x => Real.exp (-2 * x)', 'fun x => Real.exp (-x) - Real.exp (-2 * x)', 'fun x => 2 * Real.exp (-x) - Real.exp (-2 * x)', '0', '_', '_', '_', '_', '_', '_',
                    lhs=(diff(-exp(-x), x) + (-exp(-x))).subs(x=0),
                    rhs=0, relation='==', edge=True),
    oracle_instance('fun _ => 1', 'fun x => Real.exp (-2 * x)', 'fun x => Real.exp (-x) - Real.exp (-2 * x)', 'fun x => 2 * Real.exp (-x) - Real.exp (-2 * x)', '1', '_', '_', '_', '_', '_', '_',
                    lhs=(diff(-exp(-x), x) + (-exp(-x))).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun _ => 1', 'fun x => Real.exp (-2 * x)', 'fun x => Real.exp (-x) - Real.exp (-2 * x)', 'fun x => 2 * Real.exp (-x) - Real.exp (-2 * x)', '-2', '_', '_', '_', '_', '_', '_',
                    lhs=(diff(-exp(-x), x) + (-exp(-x))).subs(x=-2),
                    rhs=0, relation='==', edge=False),
]
