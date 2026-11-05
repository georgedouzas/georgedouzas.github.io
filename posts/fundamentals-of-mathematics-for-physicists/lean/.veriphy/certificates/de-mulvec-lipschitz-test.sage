# name: de-mulvec-lipschitz-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SolutionSpace.mulVec_lipschitz file "de-mulvec-lipschitz-test.sage"
# description: x ↦ Ax is Lipschitz with constant the operator norm: ‖Ax - Ay‖∞ ≤ (max row sum)‖x - y‖∞ at sample A, x, y; A = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '!![1, 2; 3, 4]',
                    lhs=max(abs(e) for e in matrix(QQ,[[1,2],[3,4]])*vector([1,-1]) - matrix(QQ,[[1,2],[3,4]])*vector([0,2])),
                    rhs=7*max(abs(e) for e in vector([1,-3])), relation='<=', edge=False),
    oracle_instance('1', '0',
                    lhs=0,
                    rhs=0, relation='<=', edge=True),
    oracle_instance('3', '!![0, 1, 0; 0, 0, 1; 0, -1, 0]',
                    lhs=max(abs(e) for e in matrix(QQ,[[0,1,0],[0,0,1],[0,-1,0]])*vector([1,2,3])),
                    rhs=1*3, relation='<=', edge=False),
]
