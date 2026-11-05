# name: set-cubic-strictmono-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.cubic_strictMono
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 ≤ a), StrictMono (fun x : ℝ => x ^ 3 + a * x) ∧ Function.Injective (fun x : ℝ => x ^ 3 + a * x) := by\n  intro a ha\n  have h : StrictMono (fun x : ℝ => x ^ 3 + a * x) := by\n    intro x y hxy\n    simp only\n    have h1 : 0 < y - x := sub_pos.2 hxy\n    have h2 : 0 < y ^ 2 + x * y + x ^ 2 + a := by nlinarith [sq_nonneg (x + y), mul_pos h1 h1]\n    nlinarith [mul_pos h1 h2]\n  exact ⟨h, h.injective⟩\nexact @key (1) ((by norm_num))'),
    (['0', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 ≤ a), StrictMono (fun x : ℝ => x ^ 3 + a * x) ∧ Function.Injective (fun x : ℝ => x ^ 3 + a * x) := by\n  intro a ha\n  have h : StrictMono (fun x : ℝ => x ^ 3 + a * x) := by\n    intro x y hxy\n    simp only\n    have h1 : 0 < y - x := sub_pos.2 hxy\n    have h2 : 0 < y ^ 2 + x * y + x ^ 2 + a := by nlinarith [sq_nonneg (x + y), mul_pos h1 h1]\n    nlinarith [mul_pos h1 h2]\n  exact ⟨h, h.injective⟩\nexact @key (0) ((by norm_num))'),
    (['2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 ≤ a), StrictMono (fun x : ℝ => x ^ 3 + a * x) ∧ Function.Injective (fun x : ℝ => x ^ 3 + a * x) := by\n  intro a ha\n  have h : StrictMono (fun x : ℝ => x ^ 3 + a * x) := by\n    intro x y hxy\n    simp only\n    have h1 : 0 < y - x := sub_pos.2 hxy\n    have h2 : 0 < y ^ 2 + x * y + x ^ 2 + a := by nlinarith [sq_nonneg (x + y), mul_pos h1 h1]\n    nlinarith [mul_pos h1 h2]\n  exact ⟨h, h.injective⟩\nexact @key (2) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
