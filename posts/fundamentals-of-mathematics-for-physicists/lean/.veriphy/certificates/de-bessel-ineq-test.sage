# name: de-bessel-ineq-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.OrthogonalFunctions.bessel_inequality file "de-bessel-ineq-test.sage"
# description: Σ⟨ψᵢ, f⟩² ≤ ‖f‖² for orthonormal families in ℝ³: the standard basis (equality), two vectors, the empty family (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 3', '_', '_', '_', 'Finset.univ',
                    lhs=1^2 + 2^2 + 2^2,
                    rhs=vector([1,2,2]).norm()^2, relation='<=', edge=False),
    oracle_instance('EuclideanSpace ℝ (Fin 3)', '_', '_', 'Fin 2', '_', '_', '_', 'Finset.univ',
                    lhs=(vector([1,2,2]).dot_product(vector([1,1,0])/sqrt(2)))^2 + 2^2,
                    rhs=vector([1,2,2]).norm()^2, relation='<=', edge=False),
    oracle_instance('ℝ', '_', '_', 'Fin 0', '_', '_', '1', '∅',
                    lhs=0,
                    rhs=1, relation='<=', edge=True),
]
