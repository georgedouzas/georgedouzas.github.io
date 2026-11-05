# name: set-angle-equivalence-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.angle_equivalence
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2 * Real.pi'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ), Equivalence (fun x y : ℝ => ∃ k : ℤ, y - x = k * T) := by\n  intro T\n  exact (⟨fun x => ⟨0, by simp⟩, fun ⟨k, hk⟩ => ⟨-k, by push_cast; linarith⟩,\n    fun ⟨k, hk⟩ ⟨l, hl⟩ => ⟨k + l, by push_cast; linarith⟩⟩)\nexact @key (2 * Real.pi)'),
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ), Equivalence (fun x y : ℝ => ∃ k : ℤ, y - x = k * T) := by\n  intro T\n  exact (⟨fun x => ⟨0, by simp⟩, fun ⟨k, hk⟩ => ⟨-k, by push_cast; linarith⟩,\n    fun ⟨k, hk⟩ ⟨l, hl⟩ => ⟨k + l, by push_cast; linarith⟩⟩)\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ), Equivalence (fun x y : ℝ => ∃ k : ℤ, y - x = k * T) := by\n  intro T\n  exact (⟨fun x => ⟨0, by simp⟩, fun ⟨k, hk⟩ => ⟨-k, by push_cast; linarith⟩,\n    fun ⟨k, hk⟩ ⟨l, hl⟩ => ⟨k + l, by push_cast; linarith⟩⟩)\nexact @key (1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
