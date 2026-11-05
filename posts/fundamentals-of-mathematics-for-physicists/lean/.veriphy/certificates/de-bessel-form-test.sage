# name: de-bessel-form-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.bessel_form file "de-bessel-form-test.sage"
# description: (xu')' = xu'' + u' by Sage for u = x², sin x, 1 (edge), at x = 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('_', '_', '2', '_',
                    lhs=(diff(x*diff(x^2, x), x) - (x*diff(x^2,x,2) + diff(x^2,x))).subs(x=2),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '2', '_',
                    lhs=(diff(x*diff(sin(x), x), x) - (x*diff(sin(x),x,2) + diff(sin(x),x))).subs(x=2),
                    rhs=0, relation='==', edge=False),
    oracle_instance('_', '_', '2', '_',
                    lhs=(diff(x*diff(SR(1), x), x) - (x*diff(SR(1),x,2) + diff(SR(1),x))).subs(x=2),
                    rhs=0, relation='==', edge=True),
]
