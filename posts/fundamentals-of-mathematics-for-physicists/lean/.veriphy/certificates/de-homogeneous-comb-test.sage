# name: de-homogeneous-comb-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.homogeneous_combination file "de-homogeneous-comb-test.sage"
# description: αy₁ + βy₂ of solutions e^{-x}, 3e^{-x} of y' + y = 0 is a solution: (α, β) = (0, 0) (edge), (1, 2), (-1/2, 5) at x = 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun _ => 1', 'fun x => Real.exp (-x)', 'fun x => 3 * Real.exp (-x)', '0', '0', '1', '_', '_',
                    lhs=(diff((0)*exp(-x) + (0)*3*exp(-x), x) + (0)*exp(-x) + (0)*3*exp(-x)).subs(x=1),
                    rhs=0, relation='==', edge=True),
    oracle_instance('fun _ => 1', 'fun x => Real.exp (-x)', 'fun x => 3 * Real.exp (-x)', '1', '2', '1', '_', '_',
                    lhs=(diff((1)*exp(-x) + (2)*3*exp(-x), x) + (1)*exp(-x) + (2)*3*exp(-x)).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun _ => 1', 'fun x => Real.exp (-x)', 'fun x => 3 * Real.exp (-x)', '-1/2', '5', '1', '_', '_',
                    lhs=(diff((-1/2)*exp(-x) + (5)*3*exp(-x), x) + (-1/2)*exp(-x) + (5)*3*exp(-x)).subs(x=1),
                    rhs=0, relation='==', edge=False),
]
