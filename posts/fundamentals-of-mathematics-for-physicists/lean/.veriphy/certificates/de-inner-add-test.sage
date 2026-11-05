# name: de-inner-add-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.inner_add_left file "de-inner-add-test.sage"
# description: linearity of ⟨·, g⟩_w by Sage's integrals at sample functions, and c = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '1', '2', 'id', 'fun x => x ^ 2', 'fun _ => 1', 'id', '_', '_', '_', '_',
                    lhs=integrate((x + 2*x^2)*x, x, 0, 1),
                    rhs=integrate(x*x, x, 0, 1) + 2*integrate(x^2*x, x, 0, 1), relation='==', edge=False),
    oracle_instance('0', 'Real.pi', '-1', 'Real.sin', 'Real.cos', 'Real.sin', 'fun _ => 1', '_', '_', '_', '_',
                    lhs=integrate((sin(x) - cos(x))*sin(x), x, 0, pi),
                    rhs=integrate(sin(x)^2, x, 0, pi) - integrate(cos(x)*sin(x), x, 0, pi), relation='==', edge=False),
    oracle_instance('0', '1', '0', 'id', 'id', 'id', 'id', '_', '_', '_', '_',
                    lhs=integrate(x*x*x, x, 0, 1),
                    rhs=integrate(x^3, x, 0, 1), relation='==', edge=True),
]
