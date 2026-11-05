# name: de-xy-taylor-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.xy_exact_taylor file "de-xy-taylor-test.sage"
# description: the k-th derivative of 2eˣ - x - 1 at 0, for k = 2 (edge), 3, 7
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '_',
                    lhs=diff(2*exp(x)-x-1, x, 2).subs(x=0),
                    rhs=2, relation='==', edge=True),
    oracle_instance('3', '_',
                    lhs=diff(2*exp(x)-x-1, x, 3).subs(x=0),
                    rhs=2, relation='==', edge=False),
    oracle_instance('7', '_',
                    lhs=diff(2*exp(x)-x-1, x, 7).subs(x=0),
                    rhs=2, relation='==', edge=False),
]
