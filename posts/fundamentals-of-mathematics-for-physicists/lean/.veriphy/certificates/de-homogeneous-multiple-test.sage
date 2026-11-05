# name: de-homogeneous-multiple-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearFirstOrder.homogeneous_is_multiple file "de-homogeneous-multiple-test.sage"
# description: μy is constant: (p, μ, y) = (1, eˣ, e^{-x}), (2x, e^{x²}, 3e^{-x²}), (0, 1, 7) (edge), at x = 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun _ => 1', 'Real.exp', 'fun x => Real.exp (-x)', '_', '_', '2',
                    lhs=(exp(x)*exp(-x)).subs(x=2),
                    rhs=exp(0)*exp(0), relation='==', edge=False),
    oracle_instance('fun x => 2 * x', 'fun x => Real.exp (x ^ 2)', 'fun x => 3 * Real.exp (-x ^ 2)', '_', '_', '2',
                    lhs=(exp(x^2)*3*exp(-x^2)).subs(x=2),
                    rhs=3, relation='==', edge=False),
    oracle_instance('fun _ => 0', 'fun _ => 1', 'fun _ => 7', '_', '_', '2',
                    lhs=7,
                    rhs=7, relation='==', edge=True),
]
