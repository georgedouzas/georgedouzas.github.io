# name: de-abs-weak-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FundamentalSolutions.abs_weak file "de-abs-weak-test.sage"
# description: ∫|x|/2 f'' = f(0) by Sage for f vanishing with f' at ±L: f = (x² - 1)², (x² - 4)² cos x, and f = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate(-x/2*diff((x^2-1)^2,x,2), x, -1, 0) + integrate(x/2*diff((x^2-1)^2,x,2), x, 0, 1),
                    rhs=1, relation='==', edge=False),
    oracle_instance('2', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_',
                    lhs=integrate(-x/2*diff((x^2-4)^2*cos(x),x,2), x, -2, 0) + integrate(x/2*diff((x^2-4)^2*cos(x),x,2), x, 0, 2),
                    rhs=16, relation='==', edge=False),
    oracle_instance('0', '_', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', '_', '_', '_', '_', '_', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
