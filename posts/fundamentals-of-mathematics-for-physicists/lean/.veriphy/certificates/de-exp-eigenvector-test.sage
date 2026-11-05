# name: de-exp-eigenvector-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_eigenvector file "de-exp-eigenvector-test.sage"
# description: e^{tA}v = e^{λt}v for eigenpairs by Sage: tanks (1, 1) with 0 (edge), tanks (1, -1) with -2, diag(1, -1) e₁ with 1, at t = 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '!![-1, 1; 1, -1]', '![1, 1]', '0', '_', '1',
                    lhs=matrix(QQ,[[-1,1],[1,-1]]).exp()*vector([1,1]),
                    rhs=vector(SR,[1,1]), relation='==', edge=True),
    oracle_instance('2', '!![-1, 1; 1, -1]', '![1, -1]', '-2', '_', '1',
                    lhs=matrix(QQ,[[-1,1],[1,-1]]).exp()*vector([1,-1]),
                    rhs=exp(-2)*vector(SR,[1,-1]), relation='==', edge=False),
    oracle_instance('2', '!![1, 0; 0, -1]', '![1, 0]', '1', '_', '1',
                    lhs=matrix(QQ,[[1,0],[0,-1]]).exp()*vector([1,0]),
                    rhs=exp(1)*vector(SR,[1,0]), relation='==', edge=False),
]
