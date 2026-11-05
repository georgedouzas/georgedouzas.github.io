# name: ca-holo-cos-zero-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.cos_eq_zero_iff'
# description: the zeros of cos at z = 0, π/2, 3π/2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0'], 'rw [Complex.cos_eq_zero_iff]\nconstructor <;> rintro ⟨k, hk⟩ <;> exact ⟨k, by linear_combination hk⟩', True),
    (['((Real.pi : ℂ) / 2)'], 'rw [Complex.cos_eq_zero_iff]\nconstructor <;> rintro ⟨k, hk⟩ <;> exact ⟨k, by linear_combination hk⟩', False),
    (['(3 * (Real.pi : ℂ) / 2)'], 'rw [Complex.cos_eq_zero_iff]\nconstructor <;> rintro ⟨k, hk⟩ <;> exact ⟨k, by linear_combination hk⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
