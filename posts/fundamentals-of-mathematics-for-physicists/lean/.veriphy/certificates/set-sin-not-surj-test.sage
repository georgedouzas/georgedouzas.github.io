# name: set-sin-not-surj-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Surjective.mul_sin_not_surjective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 ≤ a), ¬ Function.Surjective (fun x => a * Real.sin x) := by\n  intro a ha\n  intro h\n  obtain ⟨x, hx⟩ := h (a + 1)\n  nlinarith [Real.sin_le_one x]\nexact @key (1) ((by norm_num))'),
    (['0', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 ≤ a), ¬ Function.Surjective (fun x => a * Real.sin x) := by\n  intro a ha\n  intro h\n  obtain ⟨x, hx⟩ := h (a + 1)\n  nlinarith [Real.sin_le_one x]\nexact @key (0) ((by norm_num))'),
    (['3', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 ≤ a), ¬ Function.Surjective (fun x => a * Real.sin x) := by\n  intro a ha\n  intro h\n  obtain ⟨x, hx⟩ := h (a + 1)\n  nlinarith [Real.sin_le_one x]\nexact @key (3) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
