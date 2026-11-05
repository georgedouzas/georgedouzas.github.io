# name: set-neg-congruent-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.neg_congruent_iff
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), (∃ k : ℤ, -x = x + k * (2 * Real.pi)) ↔ ∃ m : ℤ, x = m * Real.pi := by\n  intro x\n  constructor\n  · rintro ⟨k, hk⟩; exact ⟨-k, by push_cast; linarith⟩\n  · rintro ⟨m, rfl⟩; exact ⟨-m, by push_cast; ring⟩\nexact @key (0)'),
    (['Real.pi'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), (∃ k : ℤ, -x = x + k * (2 * Real.pi)) ↔ ∃ m : ℤ, x = m * Real.pi := by\n  intro x\n  constructor\n  · rintro ⟨k, hk⟩; exact ⟨-k, by push_cast; linarith⟩\n  · rintro ⟨m, rfl⟩; exact ⟨-m, by push_cast; ring⟩\nexact @key (Real.pi)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), (∃ k : ℤ, -x = x + k * (2 * Real.pi)) ↔ ∃ m : ℤ, x = m * Real.pi := by\n  intro x\n  constructor\n  · rintro ⟨k, hk⟩; exact ⟨-k, by push_cast; linarith⟩\n  · rintro ⟨m, rfl⟩; exact ⟨-m, by push_cast; ring⟩\nexact @key (1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
