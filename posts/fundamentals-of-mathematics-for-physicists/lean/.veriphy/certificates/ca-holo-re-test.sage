# name: ca-holo-re-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.re_not_differentiableAt
# description: Re z is complex differentiable nowhere: z = 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'have hL : HasFDerivAt (fun w : ℂ => (w.re : ℂ)) (Complex.ofRealCLM.comp Complex.reCLM) (0) := (Complex.ofRealCLM.comp Complex.reCLM).hasFDerivAt\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]\nsimp [Complex.ext_iff]', True),
    (['1'], 'have hL : HasFDerivAt (fun w : ℂ => (w.re : ℂ)) (Complex.ofRealCLM.comp Complex.reCLM) (1) := (Complex.ofRealCLM.comp Complex.reCLM).hasFDerivAt\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]\nsimp [Complex.ext_iff]', False),
    (['Complex.I'], 'have hL : HasFDerivAt (fun w : ℂ => (w.re : ℂ)) (Complex.ofRealCLM.comp Complex.reCLM) (Complex.I) := (Complex.ofRealCLM.comp Complex.reCLM).hasFDerivAt\nrw [KaluzaKlein.ComplexAnalysis.ComplexDerivative.differentiableAt_iff_of_hasFDerivAt hL]\nsimp [Complex.ext_iff]', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
