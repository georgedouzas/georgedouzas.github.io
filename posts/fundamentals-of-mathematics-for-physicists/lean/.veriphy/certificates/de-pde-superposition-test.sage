# name: de-pde-superposition-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.pde_superposition file "de-pde-superposition-test.sage"
# description: combinations of solutions of a homogeneous linear equation are solutions: the heat operator on e^{-t} sin x and e^{-4t} sin 2x, the wave operator on sin(x - t) and cos(x + t), and α = β = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_ = var('t_')
instances = [
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '_', '_', '2', '-3', '_', '_',
                    lhs=(diff(2*exp(-t_)*sin(x) - 3*exp(-4*t_)*sin(2*x), t_) - diff(2*exp(-t_)*sin(x) - 3*exp(-4*t_)*sin(2*x), x, 2)).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '_', '_', '1/2', '5', '_', '_',
                    lhs=(diff(sin(x - t_)/2 + 5*cos(x + t_), t_, 2) - diff(sin(x - t_)/2 + 5*cos(x + t_), x, 2)).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '_', '_', '_', '_', '0', '0', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
