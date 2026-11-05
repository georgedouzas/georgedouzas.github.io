# name: de-companion-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearSystems.companion_mulVec file "de-companion-test.sage"
# description: the companion matrix shifts u and puts -Σaⱼuⱼ in the last row: n = 0 (edge), 1, 2 with sample a, u
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '![3]', '![5]', '0',
                    lhs=matrix(QQ, [[-3]])*vector([5]),
                    rhs=vector([-15]), relation='==', edge=True),
    oracle_instance('1', '![2, 3]', '![1, 4]', '0',
                    lhs=matrix(QQ, [[0, 1], [-2, -3]])*vector([1, 4]),
                    rhs=vector([4, -2*1 - 3*4]), relation='==', edge=False),
    oracle_instance('2', '![1, 0, 1]', '![1, 2, 3]', '1',
                    lhs=matrix(QQ, [[0,1,0],[0,0,1],[-1,0,-1]])*vector([1,2,3]),
                    rhs=vector([2, 3, -1-3]), relation='==', edge=False),
]
