# name: de-bvp-difference-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.bvp_difference file "de-bvp-difference-test.sage"
# description: the difference of two solutions of u'' + λu = f solves the homogeneous equation: u₁ - u₂ = C sin(√λ x) for λ = 4, f = 1 (u₁ = 1/4 + sin 2x, u₂ = 1/4), λ = 0 (edge, linear), at x = 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('4', '_', '_', '_', '_', '_', '1', '_', '_', '_', '_',
                    lhs=(diff(sin(2*x), x, 2) + 4*sin(2*x)).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('0', '_', '_', '_', '_', '_', '1', '_', '_', '_', '_',
                    lhs=diff(3*x + 1, x, 2),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1', '_', '_', '_', '_', '_', '2', '_', '_', '_', '_',
                    lhs=(diff(5*cos(x), x, 2) + 5*cos(x)).subs(x=2),
                    rhs=0, relation='==', edge=False),
]
