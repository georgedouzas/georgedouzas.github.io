# name: de-tent-weak-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.tent_weak file "de-tent-weak-test.sage"
# description: -∫₀¹ G(x, ξ)f''(x)dx = f(ξ) by Sage's integrals for f = x(1 - x), sin πx, x²(1 - x) at ξ = 1/3, 1/2, and f = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1/3', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=-(integrate(x*(1-1/3)*diff(x*(1-x),x,2), x, 0, 1/3) + integrate(1/3*(1-x)*diff(x*(1-x),x,2), x, 1/3, 1)),
                    rhs=1/3*(1-1/3), relation='==', edge=False),
    oracle_instance('1/2', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=-(integrate(x*(1-1/2)*diff(sin(pi*x),x,2), x, 0, 1/2) + integrate(1/2*(1-x)*diff(sin(pi*x),x,2), x, 1/2, 1)),
                    rhs=sin(pi/2), relation='==', edge=False),
    oracle_instance('0', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', '_', '_', '_', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
