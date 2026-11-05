# name: de-eigenvalue-real-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.eigenvalue_real file "de-eigenvalue-real-test.sage"
# description: (λ - λ̄)W = 0 with W > 0 forces λ real: λ = 2 (edge), 3 - 0i, and the non-real λ = i gives (λ - λ̄)W ≠ 0
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '1', '_', '_',
                    lhs=(2 - conjugate(2))*1,
                    rhs=0, relation='==', edge=True),
    oracle_instance('3', '5', '_', '_',
                    lhs=(3 - conjugate(3))*5,
                    rhs=0, relation='==', edge=False),
    oracle_instance('Complex.I', '1', '_', '_',
                    lhs=B((I - conjugate(I))*1 != 0),
                    rhs=1, relation='==', edge=False),
]
