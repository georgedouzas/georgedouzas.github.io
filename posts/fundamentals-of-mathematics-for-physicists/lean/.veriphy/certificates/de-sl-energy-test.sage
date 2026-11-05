# name: de-sl-energy-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.sl_energy_identity file "de-sl-energy-test.sage"
# description: ∫ₐᵇ -P'u = ∫ₐᵇ Pu' for u vanishing at the ends: (P, u) = (u', sin x) on [0, π], (1 + x)u' with u = x(1 - x) on [0, 1], u = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', 'Real.pi', 'Real.cos', '_', 'Real.sin', 'Real.cos', '_', '_', '_', '_', '_', '_',
                    lhs=integrate(-diff(cos(x),x)*sin(x), x, 0, pi),
                    rhs=integrate(cos(x)*cos(x), x, 0, pi), relation='==', edge=False),
    oracle_instance('0', '1', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate(-diff((1+x)*(1-2*x),x)*x*(1-x), x, 0, 1),
                    rhs=integrate((1+x)*(1-2*x)*(1-2*x), x, 0, 1), relation='==', edge=False),
    oracle_instance('0', '1', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', '_', '_', '_', '_', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
