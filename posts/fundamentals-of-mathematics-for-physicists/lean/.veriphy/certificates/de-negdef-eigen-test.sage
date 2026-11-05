# name: de-negdef-eigen-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.negdef_eigenvalue file "de-negdef-eigen-test.sage"
# description: an eigenvector with v·Av < 0 has λ < 0: (-2, 1; 1, -2) with (1, 1), λ = -1 and (1, -1), λ = -3; (-1) with (1) (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '!![-2, 1; 1, -2]', '![1, 1]', '-1', '_', '_', '_',
                    lhs=-1,
                    rhs=0, relation='<', edge=False),
    oracle_instance('2', '!![-2, 1; 1, -2]', '![1, -1]', '-3', '_', '_', '_',
                    lhs=vector([1,-1]).dot_product(matrix(QQ,[[-2,1],[1,-2]])*vector([1,-1]))/2,
                    rhs=0, relation='<', edge=False),
    oracle_instance('1', '!![-1]', '![1]', '-1', '_', '_', '_',
                    lhs=-1,
                    rhs=0, relation='<', edge=True),
]
