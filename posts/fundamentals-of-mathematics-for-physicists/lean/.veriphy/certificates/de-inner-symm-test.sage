# name: de-inner-symm-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.inner_symm file "de-inner-symm-test.sage"
# description: ⟨f, g⟩_w = ⟨g, f⟩_w by Sage's integrals: (f, g, w) = (x, x², 1) on [0, 1], (sin, cos, x) on [0, π], (0, 1, 1) (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '1', 'id', 'fun x => x ^ 2', 'fun _ => 1',
                    lhs=integrate(x*x^2, x, 0, 1),
                    rhs=integrate(x^2*x, x, 0, 1), relation='==', edge=False),
    oracle_instance('0', 'Real.pi', 'Real.sin', 'Real.cos', 'id',
                    lhs=integrate(sin(x)*cos(x)*x, x, 0, pi),
                    rhs=integrate(cos(x)*sin(x)*x, x, 0, pi), relation='==', edge=False),
    oracle_instance('0', '1', 'fun _ => 0', 'fun _ => 1', 'fun _ => 1',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
