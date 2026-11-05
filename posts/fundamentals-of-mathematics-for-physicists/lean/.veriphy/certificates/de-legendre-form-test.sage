# name: de-legendre-form-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.legendre_form file "de-legendre-form-test.sage"
# description: ((1 - x²)u')' = (1 - x²)u'' - 2xu' by Sage for u = x³, eˣ, 1 (edge), at x = 1/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('_', '_', '_', '1/2', '_',
                    lhs=(diff((1-x^2)*diff(x^3, x), x) - ((1-x^2)*diff(x^3,x,2) - 2*x*diff(x^3,x))).subs(x=1/2),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '_', '1/2', '_',
                    lhs=(diff((1-x^2)*diff(exp(x), x), x) - ((1-x^2)*diff(exp(x),x,2) - 2*x*diff(exp(x),x))).subs(x=1/2),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '_', '1/2', '_',
                    lhs=(diff((1-x^2)*diff(SR(1), x), x) - ((1-x^2)*diff(SR(1),x,2) - 2*x*diff(SR(1),x))).subs(x=1/2),
                    rhs=0, relation='==', edge=True),
]
