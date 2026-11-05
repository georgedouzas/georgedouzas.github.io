# name: set-tan-bijon-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Bijective.mul_tan_bijOn
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => a * Real.tan x) (Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) Set.univ := by\n  intro a ha\n  refine ⟨fun _ _ => Set.mem_univ _, fun x hx y hy h => ?_, fun y _ => ?_⟩\n  · have h' : Real.tan x = Real.tan y := mul_left_cancel₀ ha h\n    exact Real.strictMonoOn_tan.injOn hx hy h'\n  · refine ⟨Real.arctan (y / a), ⟨Real.neg_pi_div_two_lt_arctan _, Real.arctan_lt_pi_div_two _⟩, ?_⟩\n    simp only\n    rw [Real.tan_arctan]; field_simp\nexact @key (1) ((by norm_num))"),
    (['2', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => a * Real.tan x) (Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) Set.univ := by\n  intro a ha\n  refine ⟨fun _ _ => Set.mem_univ _, fun x hx y hy h => ?_, fun y _ => ?_⟩\n  · have h' : Real.tan x = Real.tan y := mul_left_cancel₀ ha h\n    exact Real.strictMonoOn_tan.injOn hx hy h'\n  · refine ⟨Real.arctan (y / a), ⟨Real.neg_pi_div_two_lt_arctan _, Real.arctan_lt_pi_div_two _⟩, ?_⟩\n    simp only\n    rw [Real.tan_arctan]; field_simp\nexact @key (2) ((by norm_num))"),
    (['-1', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : a ≠ 0), Set.BijOn (fun x => a * Real.tan x) (Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) Set.univ := by\n  intro a ha\n  refine ⟨fun _ _ => Set.mem_univ _, fun x hx y hy h => ?_, fun y _ => ?_⟩\n  · have h' : Real.tan x = Real.tan y := mul_left_cancel₀ ha h\n    exact Real.strictMonoOn_tan.injOn hx hy h'\n  · refine ⟨Real.arctan (y / a), ⟨Real.neg_pi_div_two_lt_arctan _, Real.arctan_lt_pi_div_two _⟩, ?_⟩\n    simp only\n    rw [Real.tan_arctan]; field_simp\nexact @key (-1) ((by norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
