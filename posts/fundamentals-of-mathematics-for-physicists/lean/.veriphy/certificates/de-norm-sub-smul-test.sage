# name: de-norm-sub-smul-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.norm_sub_smul_sq file "de-norm-sub-smul-test.sage"
# description: ‖f - cφ‖² = ‖f‖² - 2c⟨f, φ⟩ + c²‖φ‖² in ℝ² and ℝ³ with exact vectors, and c = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('EuclideanSpace ℝ (Fin 2)', '_', '_', '_', '_', '2',
                    lhs=(vector([1,2]) - 2*vector([3,-1])).norm()^2,
                    rhs=vector([1,2]).norm()^2 - 2*2*vector([1,2]).dot_product(vector([3,-1])) + 4*vector([3,-1]).norm()^2, relation='==', edge=False),
    oracle_instance('EuclideanSpace ℝ (Fin 3)', '_', '_', '_', '_', '-1/2',
                    lhs=(vector([1,0,2]) + 1/2*vector([0,1,1])).norm()^2,
                    rhs=vector([1,0,2]).norm()^2 + vector([1,0,2]).dot_product(vector([0,1,1])) + 1/4*vector([0,1,1]).norm()^2, relation='==', edge=False),
    oracle_instance('ℝ', '_', '_', '3', '5', '0',
                    lhs=9,
                    rhs=9, relation='==', edge=True),
]
