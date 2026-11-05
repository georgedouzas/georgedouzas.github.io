# name: ca-holo-conj-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.conj_not_differentiableAt
# description: conjugation is real but not complex differentiable at 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'have hL : HasFDerivAt (fun w : ℂ => (starRingEnd ℂ) w) (Complex.conjCLE : ℂ →L[ℝ] ℂ) (0) := Complex.conjCLE.hasFDerivAt\nrefine ⟨hL.differentiableAt, ?_⟩\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]\nnorm_num [Complex.ext_iff]', True),
    (['1'], 'have hL : HasFDerivAt (fun w : ℂ => (starRingEnd ℂ) w) (Complex.conjCLE : ℂ →L[ℝ] ℂ) (1) := Complex.conjCLE.hasFDerivAt\nrefine ⟨hL.differentiableAt, ?_⟩\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]\nnorm_num [Complex.ext_iff]', False),
    (['Complex.I'], 'have hL : HasFDerivAt (fun w : ℂ => (starRingEnd ℂ) w) (Complex.conjCLE : ℂ →L[ℝ] ℂ) (Complex.I) := Complex.conjCLE.hasFDerivAt\nrefine ⟨hL.differentiableAt, ?_⟩\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]\nnorm_num [Complex.ext_iff]', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
