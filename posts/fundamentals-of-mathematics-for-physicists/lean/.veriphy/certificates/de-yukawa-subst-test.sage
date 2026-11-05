# name: de-yukawa-subst-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.GreensComputations.yukawa_substitution_derivative file "de-yukawa-subst-test.sage"
# description: (v/s)' = (rv' - v)/r² by Sage for v = e^{-r}, sin r, 1 (edge), at r = 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('_', '_', '2', '_', '_',
                    lhs=diff((exp(-x))/x, x).subs(x=2),
                    rhs=((2*diff(exp(-x), x) - (exp(-x)))/x^2).subs(x=2), relation='==', edge=False),
    oracle_instance('_', '_', '2', '_', '_',
                    lhs=diff((sin(x))/x, x).subs(x=2),
                    rhs=((2*diff(sin(x), x) - (sin(x)))/x^2).subs(x=2), relation='==', edge=False),
    oracle_instance('_', '_', '2', '_', '_',
                    lhs=diff((SR(1))/x, x).subs(x=2),
                    rhs=((2*diff(SR(1), x) - (SR(1)))/x^2).subs(x=2), relation='==', edge=True),
]
