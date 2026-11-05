# name: ca-holo-cos-real-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_zero_real
# description: the zeros π/2 + kπ of cos are real, k = 0, 1, -1; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['((Real.pi : ℂ) / 2 + ((0 : ℤ) : ℂ) * (Real.pi : ℂ))', '(by rw [Complex.cos_eq_zero_iff]; exact ⟨0, by push_cast; ring⟩)'], 'simp', True),
    (['((Real.pi : ℂ) / 2 + ((1 : ℤ) : ℂ) * (Real.pi : ℂ))', '(by rw [Complex.cos_eq_zero_iff]; exact ⟨1, by push_cast; ring⟩)'], 'simp', False),
    (['((Real.pi : ℂ) / 2 + ((-1 : ℤ) : ℂ) * (Real.pi : ℂ))', '(by rw [Complex.cos_eq_zero_iff]; exact ⟨-1, by push_cast; ring⟩)'], 'simp', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
