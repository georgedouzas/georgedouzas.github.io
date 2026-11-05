# name: de-J-powers-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearSystemComputations.J_powers file "de-J-powers-test.sage"
# description: J^{2k} = (-1)^k and J^{2k+1} = (-1)^kJ by Sage at k = 0 (edge), 1, 5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_ = var('t_')
A2 = matrix(QQ, [[0, 1], [-2, -2]])
J = matrix(QQ, [[0, 1], [-1, 0]])
N = matrix(QQ, [[0, 1], [0, 0]])
instances = [
    oracle_instance('0',
                    lhs=B(J^(2*0) == (-1)^0*identity_matrix(QQ,2) and J^(2*0+1) == (-1)^0*J),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1',
                    lhs=B(J^(2*1) == (-1)^1*identity_matrix(QQ,2) and J^(2*1+1) == (-1)^1*J),
                    rhs=1, relation='==', edge=False),
    oracle_instance('5',
                    lhs=B(J^(2*5) == (-1)^5*identity_matrix(QQ,2) and J^(2*5+1) == (-1)^5*J),
                    rhs=1, relation='==', edge=False),
]
