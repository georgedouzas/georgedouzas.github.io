# name: ca-holo-normsq-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_differentiableAt_iff
# description: |z|^2 is complex differentiable at 0 only: z = 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'refine ⟨?_, KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_hasDerivAt_zero⟩\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt (KaluzaKlein.ComplexAnalysis.ComplexDerivative.hasFDerivAt_normSq (0))]\nsimp [Complex.ext_iff]', True),
    (['1'], 'refine ⟨?_, KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_hasDerivAt_zero⟩\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt (KaluzaKlein.ComplexAnalysis.ComplexDerivative.hasFDerivAt_normSq (1))]\nsimp [Complex.ext_iff]', False),
    (['Complex.I'], 'refine ⟨?_, KaluzaKlein.ComplexAnalysis.ComplexDerivative.normSq_hasDerivAt_zero⟩\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt (KaluzaKlein.ComplexAnalysis.ComplexDerivative.hasFDerivAt_normSq (Complex.I))]\nsimp [Complex.ext_iff]', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
