# name: de-legendre-orthonormal-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.legendre_orthonormal file "de-legendre-orthonormal-test.sage"
# description: ((2n+1)/2)(2/(2n+1)) = 1 at n = 0 (edge), 3, 10
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=((2*0+1)/2)*(2/(2*0+1)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('3',
                    lhs=((2*3+1)/2)*(2/(2*3+1)),
                    rhs=1, relation='==', edge=False),
    oracle_instance('10',
                    lhs=((2*10+1)/2)*(2/(2*10+1)),
                    rhs=1, relation='==', edge=False),
]
