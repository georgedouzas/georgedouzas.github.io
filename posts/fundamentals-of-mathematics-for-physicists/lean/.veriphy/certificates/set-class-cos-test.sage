# name: set-class-cos-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.class_cos
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), {y : ℝ | Real.cos y = Real.cos x} = {y | ∃ k : ℤ, y = x + k * (2 * Real.pi) ∨ y = -x + k * (2 * Real.pi)} := by\n  intro x\n  ext y\n  simp only [Set.mem_setOf_eq, Real.cos_eq_cos_iff]\n  constructor\n  · rintro ⟨k, h | h⟩\n    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩\n    · exact ⟨k, Or.inr (by linarith)⟩\n  · rintro ⟨k, h | h⟩\n    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩\n    · exact ⟨k, Or.inr (by linarith)⟩\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), {y : ℝ | Real.cos y = Real.cos x} = {y | ∃ k : ℤ, y = x + k * (2 * Real.pi) ∨ y = -x + k * (2 * Real.pi)} := by\n  intro x\n  ext y\n  simp only [Set.mem_setOf_eq, Real.cos_eq_cos_iff]\n  constructor\n  · rintro ⟨k, h | h⟩\n    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩\n    · exact ⟨k, Or.inr (by linarith)⟩\n  · rintro ⟨k, h | h⟩\n    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩\n    · exact ⟨k, Or.inr (by linarith)⟩\nexact @key (1)'),
    (['Real.pi'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), {y : ℝ | Real.cos y = Real.cos x} = {y | ∃ k : ℤ, y = x + k * (2 * Real.pi) ∨ y = -x + k * (2 * Real.pi)} := by\n  intro x\n  ext y\n  simp only [Set.mem_setOf_eq, Real.cos_eq_cos_iff]\n  constructor\n  · rintro ⟨k, h | h⟩\n    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩\n    · exact ⟨k, Or.inr (by linarith)⟩\n  · rintro ⟨k, h | h⟩\n    · exact ⟨-k, Or.inl (by push_cast; linarith)⟩\n    · exact ⟨k, Or.inr (by linarith)⟩\nexact @key (Real.pi)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
