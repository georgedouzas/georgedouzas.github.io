# name: de-symmetric-orth-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Eigenvalues.symmetric_eigenvectors_orthogonal file "de-symmetric-orth-test.sage"
# description: eigenvectors of a symmetric matrix for different eigenvalues are orthogonal: (1, 2; 2, 1) with (1, 1), (1, -1); diag(1, 0) with e₁, e₂ (edge); (2, 1, 0; …) with Sage's eigenvectors
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '!![1, 2; 2, 1]', '_', '![1, 1]', '![1, -1]', '3', '-1', '_', '_', '_',
                    lhs=vector([1,1]).dot_product(vector([1,-1])),
                    rhs=0, relation='==', edge=False),
    oracle_instance('2', '!![1, 0; 0, 0]', '_', '![1, 0]', '![0, 1]', '1', '0', '_', '_', '_',
                    lhs=vector([1,0]).dot_product(vector([0,1])),
                    rhs=0, relation='==', edge=True),
    oracle_instance('3', '!![2, 1, 0; 1, 2, 1; 0, 1, 2]', '_', '![1, 0, -1]', '![1, Real.sqrt 2, 1]', '2', '2 + Real.sqrt 2', '_', '_', '_',
                    lhs=vector(SR,[1,0,-1]).dot_product(vector(SR,[1,sqrt(2),1])),
                    rhs=0, relation='==', edge=False),
]
