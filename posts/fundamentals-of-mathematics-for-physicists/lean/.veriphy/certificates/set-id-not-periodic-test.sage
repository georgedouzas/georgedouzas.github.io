# name: set-id-not-periodic-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Quotients.id_not_periodic
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2 * Real.pi', '(by positivity)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ) (hT : T ≠ 0), ¬ Function.Periodic (fun φ : ℝ => φ) T := by\n  intro T hT\n  intro h\n  have := h 0\n  simp at this\n  exact hT this\nexact @key (2 * Real.pi) ((by positivity))'),
    (['1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ) (hT : T ≠ 0), ¬ Function.Periodic (fun φ : ℝ => φ) T := by\n  intro T hT\n  intro h\n  have := h 0\n  simp at this\n  exact hT this\nexact @key (1) ((by norm_num))'),
    (['-3', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ) (hT : T ≠ 0), ¬ Function.Periodic (fun φ : ℝ => φ) T := by\n  intro T hT\n  intro h\n  have := h 0\n  simp at this\n  exact hT this\nexact @key (-3) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
