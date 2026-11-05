# name: de-eigen-orthogonal-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.eigen_orthogonal file "de-eigen-orthogonal-test.sage"
# description: ∫ₐᵇ g = 0 when g = F' with F vanishing at the ends: F = sin x sin 2x on [0, π], F = x(1 - x) on [0, 1], F = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', 'Real.pi', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate(diff(sin(x)*sin(2*x), x), x, 0, pi),
                    rhs=0, relation='==', edge=False),
    oracle_instance('0', '1', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate(diff(x*(1-x), x), x, 0, 1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('0', '1', '_', '_', 'fun _ => 0', 'fun _ => 0', '_', '_', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
