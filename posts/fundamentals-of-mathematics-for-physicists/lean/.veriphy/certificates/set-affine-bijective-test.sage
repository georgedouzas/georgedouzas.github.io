# name: set-affine-bijective-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Bijective.affine_bijective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2', '1', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (ha : a ≠ 0), Function.Bijective (fun x : ℝ => a * x + b) := by\n  intro a b ha\n  exact (⟨fun x y h => mul_left_cancel₀ ha (by simpa using h), fun y => ⟨(y - b) / a, by field_simp; ring⟩⟩)\nexact @key (2) (1) ((by norm_num))'),
    (['1', '0', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (ha : a ≠ 0), Function.Bijective (fun x : ℝ => a * x + b) := by\n  intro a b ha\n  exact (⟨fun x y h => mul_left_cancel₀ ha (by simpa using h), fun y => ⟨(y - b) / a, by field_simp; ring⟩⟩)\nexact @key (1) (0) ((by norm_num))'),
    (['-3', '5', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (ha : a ≠ 0), Function.Bijective (fun x : ℝ => a * x + b) := by\n  intro a b ha\n  exact (⟨fun x y h => mul_left_cancel₀ ha (by simpa using h), fun y => ⟨(y - b) / a, by field_simp; ring⟩⟩)\nexact @key (-3) (5) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
