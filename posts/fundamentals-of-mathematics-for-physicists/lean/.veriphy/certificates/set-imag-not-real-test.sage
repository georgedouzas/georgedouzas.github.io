# name: set-imag-not-real-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.imag_not_real
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (y : ℝ) (hy : y ≠ 0), ¬ ∃ x : ℝ, (x : ℂ) = y * Complex.I := by\n  intro y hy\n  rintro ⟨x, hx⟩\n  have := congrArg Complex.im hx\n  simp at this\n  exact hy this.symm\nexact @key (1) ((by norm_num))'),
    (['-2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (y : ℝ) (hy : y ≠ 0), ¬ ∃ x : ℝ, (x : ℂ) = y * Complex.I := by\n  intro y hy\n  rintro ⟨x, hx⟩\n  have := congrArg Complex.im hx\n  simp at this\n  exact hy this.symm\nexact @key (-2) ((by norm_num))'),
    (['1/2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (y : ℝ) (hy : y ≠ 0), ¬ ∃ x : ℝ, (x : ℂ) = y * Complex.I := by\n  intro y hy\n  rintro ⟨x, hx⟩\n  have := congrArg Complex.im hx\n  simp at this\n  exact hy this.symm\nexact @key (1/2) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
