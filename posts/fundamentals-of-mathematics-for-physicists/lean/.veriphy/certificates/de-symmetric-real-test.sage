# name: de-symmetric-real-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Eigenvalues.symmetric_eigenvalue_real file "de-symmetric-real-test.sage"
# description: the eigenvalues of real symmetric matrices are real: Sage's eigenvalues of three symmetric matrices, one the identity (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '1', '_', '_', '_', '1', '_',
                    lhs=B(all(e.imag() == 0 for e in identity_matrix(QQ,2).eigenvalues())),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '!![1, 2; 2, 1]', '_', '_', '_', '3', '_',
                    lhs=B(all(QQbar(e).imag() == 0 for e in matrix(QQ,[[1,2],[2,1]]).eigenvalues())),
                    rhs=1, relation='==', edge=False),
    oracle_instance('3', '!![2, 1, 0; 1, 2, 1; 0, 1, 2]', '_', '_', '_', '2', '_',
                    lhs=B(all(QQbar(e).imag() == 0 for e in matrix(QQ,[[2,1,0],[1,2,1],[0,1,2]]).eigenvalues())),
                    rhs=1, relation='==', edge=False),
]
