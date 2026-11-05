# name: de-integrating-solution-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.integrating_factor_solution file "de-integrating-solution-test.sage"
# description: y = (M + C)/μ solves y' + py = q: (p, q, C) = (2, e^{-x}, 1), (1, 1, 0) (edge), (x, x, 2), at x = 1/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun x => Real.exp (2 * x)', 'Real.exp', 'fun _ => 2', 'fun x => Real.exp (-x)', '1', '1/2', '_', '_', '_',
                    lhs=(diff((exp(x)+1)/exp(2*x), x) + 2*(exp(x)+1)/exp(2*x)).simplify_full().subs(x=1/2),
                    rhs=exp(-1/2), relation='==', edge=False),
    oracle_instance('Real.exp', 'Real.exp', 'fun _ => 1', 'fun _ => 1', '0', '1/2', '_', '_', '_',
                    lhs=(diff(exp(x)/exp(x), x) + exp(x)/exp(x)).simplify_full(),
                    rhs=1, relation='==', edge=True),
    oracle_instance('fun x => Real.exp (x ^ 2 / 2)', 'fun x => Real.exp (x ^ 2 / 2)', 'id', 'id', '2', '1/2', '_', '_', '_',
                    lhs=(diff((exp(x^2/2)+2)/exp(x^2/2), x) + x*(exp(x^2/2)+2)/exp(x^2/2)).simplify_full().subs(x=1/2),
                    rhs=1/2, relation='==', edge=False),
]
