# name: de-similarity-exact-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HeatEquation.similarity_exact file "de-similarity-exact-test.sage"
# description: (F' + 2ξF)' = F'' + 2ξF' + 2F by Sage for generic F, and F = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
FF = function('FF')
instances = [
    oracle_instance('_', '_', '_', 'x', '_', '_',
                    lhs=diff(diff(FF(x), x) + 2*x*FF(x), x),
                    rhs=diff(FF(x), x, 2) + 2*x*diff(FF(x), x) + 2*FF(x), relation='==', edge=False),
    oracle_instance('_', '_', '_', '1', '_', '_',
                    lhs=diff(diff(exp(-x^2), x) + 2*x*exp(-x^2), x).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun _ => 0', 'fun _ => 0', 'fun _ => 0', '0', '_', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
