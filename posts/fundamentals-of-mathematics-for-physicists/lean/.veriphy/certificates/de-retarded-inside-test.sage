# name: de-retarded-inside-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.RetardedGreens.retarded_1d_inside file "de-retarded-inside-test.sage"
# description: ½∫_{x-t}^{x+t}ψ = ½ inside the cone for a bump ψ of unit integral on (-ε, ε): Sage's integrals for ψ = (3/4ε)(1 - y²/ε²) at (ε, t, x) = (1/2, 2, 1/2), (1, 3, -1), (1/10, 1, 0) (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
y_ = var('y_')
instances = [
    oracle_instance('_', '1/2', '2', '1/2', '_', '_', '_', '_',
                    lhs=1/2*integrate(3/(4*(1/2))*(1 - y_^2/(1/2)^2), y_, -(1/2), 1/2),
                    rhs=1/2, relation='==', edge=False),
    oracle_instance('_', '1', '3', '-1', '_', '_', '_', '_',
                    lhs=1/2*integrate(3/(4*(1))*(1 - y_^2/(1)^2), y_, -(1), 1),
                    rhs=1/2, relation='==', edge=False),
    oracle_instance('_', '1/10', '1', '0', '_', '_', '_', '_',
                    lhs=1/2*integrate(3/(4*(1/10))*(1 - y_^2/(1/10)^2), y_, -(1/10), 1/10),
                    rhs=1/2, relation='==', edge=True),
]
