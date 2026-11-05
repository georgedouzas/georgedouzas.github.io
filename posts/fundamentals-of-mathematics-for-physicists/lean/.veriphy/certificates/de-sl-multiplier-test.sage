# name: de-sl-multiplier-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.sl_multiplier file "de-sl-multiplier-test.sage"
# description: (pu')' = (p/a)(au'' + bu') when p' = (b/a)p: Hermite (a = 1, b = -2x, p = e^{-x²}), (a, b) = (x², x) with p = x, a = 1, b = 0 (edge), for u = x³ at x = 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun _ => 1', 'fun x => -2 * x', 'fun x => Real.exp (-x ^ 2)', '_', '_', '1', '_', '_', '_',
                    lhs=(diff(exp(-x^2)*diff(x^3,x), x) - exp(-x^2)*(diff(x^3,x,2) - 2*x*diff(x^3,x))).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun x => x ^ 2', 'id', 'id', '_', '_', '1', '_', '_', '_',
                    lhs=(diff(x*diff(x^3,x), x) - (x/x^2)*(x^2*diff(x^3,x,2) + x*diff(x^3,x))).subs(x=1),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun _ => 1', 'fun _ => 0', 'fun _ => 1', '_', '_', '1', '_', '_', '_',
                    lhs=(diff(diff(x^3,x), x) - diff(x^3,x,2)).subs(x=1),
                    rhs=0, relation='==', edge=True),
]
