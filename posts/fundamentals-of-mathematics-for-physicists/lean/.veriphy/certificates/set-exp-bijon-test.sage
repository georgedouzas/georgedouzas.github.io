# name: set-exp-bijon-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Bijective.mul_exp_bijOn
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), Set.BijOn (fun x => a * Real.exp x) Set.univ (Set.Ioi 0) := by\n  intro a ha\n  refine ⟨fun x _ => mul_pos ha (Real.exp_pos x), fun x _ y _ h => ?_, fun y hy => ?_⟩\n  · exact Real.exp_injective (mul_left_cancel₀ ha.ne' h)\n  · obtain ⟨x, hx⟩ := (Surjective.range_mul_exp a ha).symm ▸ hy\n    exact ⟨x, Set.mem_univ _, hx⟩\nexact @key (1) ((by norm_num))"),
    (['2', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), Set.BijOn (fun x => a * Real.exp x) Set.univ (Set.Ioi 0) := by\n  intro a ha\n  refine ⟨fun x _ => mul_pos ha (Real.exp_pos x), fun x _ y _ h => ?_, fun y hy => ?_⟩\n  · exact Real.exp_injective (mul_left_cancel₀ ha.ne' h)\n  · obtain ⟨x, hx⟩ := (Surjective.range_mul_exp a ha).symm ▸ hy\n    exact ⟨x, Set.mem_univ _, hx⟩\nexact @key (2) ((by norm_num))"),
    (['1/2', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), Set.BijOn (fun x => a * Real.exp x) Set.univ (Set.Ioi 0) := by\n  intro a ha\n  refine ⟨fun x _ => mul_pos ha (Real.exp_pos x), fun x _ y _ h => ?_, fun y hy => ?_⟩\n  · exact Real.exp_injective (mul_left_cancel₀ ha.ne' h)\n  · obtain ⟨x, hx⟩ := (Surjective.range_mul_exp a ha).symm ▸ hy\n    exact ⟨x, Set.mem_univ _, hx⟩\nexact @key (1/2) ((by norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
