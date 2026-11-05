# name: set-range-exp-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Surjective.range_mul_exp
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), Set.range (fun x => a * Real.exp x) = Set.Ioi 0 := by\n  intro a ha\n  ext y\n  simp only [Set.mem_range, Set.mem_Ioi]\n  constructor\n  · rintro ⟨x, rfl⟩; positivity\n  · intro hy\n    refine ⟨Real.log (y / a), ?_⟩\n    rw [Real.exp_log (div_pos hy ha)]; field_simp\nexact @key (1) ((by norm_num))'),
    (['2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), Set.range (fun x => a * Real.exp x) = Set.Ioi 0 := by\n  intro a ha\n  ext y\n  simp only [Set.mem_range, Set.mem_Ioi]\n  constructor\n  · rintro ⟨x, rfl⟩; positivity\n  · intro hy\n    refine ⟨Real.log (y / a), ?_⟩\n    rw [Real.exp_log (div_pos hy ha)]; field_simp\nexact @key (2) ((by norm_num))'),
    (['1/2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), Set.range (fun x => a * Real.exp x) = Set.Ioi 0 := by\n  intro a ha\n  ext y\n  simp only [Set.mem_range, Set.mem_Ioi]\n  constructor\n  · rintro ⟨x, rfl⟩; positivity\n  · intro hy\n    refine ⟨Real.log (y / a), ?_⟩\n    rw [Real.exp_log (div_pos hy ha)]; field_simp\nexact @key (1/2) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
