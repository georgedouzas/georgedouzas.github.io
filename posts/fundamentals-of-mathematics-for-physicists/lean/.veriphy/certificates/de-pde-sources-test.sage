# name: de-pde-sources-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.superposition_sources file "de-pde-sources-test.sage"
# description: u₁ + u₂ solves Lu = f₁ + f₂: L = d²/dx² with (u₁, f₁) = (x², 2), (u₂, f₂) = (sin x, -sin x); (eˣ, eˣ) and (0, 0) (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=diff(x^2 + sin(x), x, 2),
                    rhs=2 - sin(x), relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=diff(exp(x) + x^3, x, 2),
                    rhs=exp(x) + 6*x, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '0', '0', '0', '0', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
