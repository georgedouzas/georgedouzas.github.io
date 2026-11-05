# name: de-negdef-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.negdef_bound file "de-negdef-test.sage"
# description: a negative definite form has v·Av ≤ -c|v|² with c the least |eigenvalue|: Sage's eigenvalues of -1, (-2, 1; 1, -2), diag(-1, -3) (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '!![-1]', '_',
                    lhs=max(matrix(QQ,[[-1]]).eigenvalues()),
                    rhs=0, relation='<', edge=False),
    oracle_instance('2', '!![-2, 1; 1, -2]', '_',
                    lhs=max(matrix(QQ,[[-2,1],[1,-2]]).eigenvalues()),
                    rhs=0, relation='<', edge=False),
    oracle_instance('2', '!![-1, 0; 0, -3]', '_',
                    lhs=max(diagonal_matrix(QQ,[-1,-3]).eigenvalues()),
                    rhs=0, relation='<', edge=True),
]
