# name: set-half-odd-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.half_odd_not_int
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (m : ℤ), ¬ ∃ k : ℤ, (k : ℚ) = (2 * m + 1) / 2 := by\n  intro m\n  rintro ⟨k, hk⟩\n  have h : ((2 * k : ℤ) : ℚ) = ((2 * m + 1 : ℤ) : ℚ) := by push_cast; linarith\n  have h' : 2 * k = 2 * m + 1 := by exact_mod_cast h\n  omega\nexact @key (0)"),
    (['1'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (m : ℤ), ¬ ∃ k : ℤ, (k : ℚ) = (2 * m + 1) / 2 := by\n  intro m\n  rintro ⟨k, hk⟩\n  have h : ((2 * k : ℤ) : ℚ) = ((2 * m + 1 : ℤ) : ℚ) := by push_cast; linarith\n  have h' : 2 * k = 2 * m + 1 := by exact_mod_cast h\n  omega\nexact @key (1)"),
    (['-3'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (m : ℤ), ¬ ∃ k : ℤ, (k : ℚ) = (2 * m + 1) / 2 := by\n  intro m\n  rintro ⟨k, hk⟩\n  have h : ((2 * k : ℤ) : ℚ) = ((2 * m + 1 : ℤ) : ℚ) := by push_cast; linarith\n  have h' : 2 * k = 2 * m + 1 := by exact_mod_cast h\n  omega\nexact @key (-3)"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
