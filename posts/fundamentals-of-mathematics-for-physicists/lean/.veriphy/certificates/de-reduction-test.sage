# name: de-reduction-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrdinaryDifferentialEquations.reduction_to_first_order file "de-reduction-test.sage"
# description: y = e^{2x}, y'' = 4y (n = 1, f(x, u) = 4u₀), y = sin (y'' = -y), y = x³ (n = 2, y''' = 6): the system's components at x = 0, 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', 'fun _ u => 4 * u 0', 'fun x => Real.exp (2 * x)', '_', '0',
                    lhs=B(diff(exp(2*x), x).subs(x=0) == diff(exp(2*x), x).subs(x=0) and diff(exp(2*x), x, 2).subs(x=0) == 4*exp(0)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1', 'fun _ u => -u 0', 'Real.sin', '_', '1',
                    lhs=B(bool(diff(sin(x), x, 2).subs(x=1) == -sin(1))),
                    rhs=1, relation='==', edge=False),
    oracle_instance('2', 'fun _ _ => 6', 'fun x => x ^ 3', '_', '1',
                    lhs=B(diff(x^3, x, 3).subs(x=1) == 6 and diff(x^3, x).subs(x=1) == 3 and diff(x^3, x, 2).subs(x=1) == 6),
                    rhs=1, relation='==', edge=False),
]
