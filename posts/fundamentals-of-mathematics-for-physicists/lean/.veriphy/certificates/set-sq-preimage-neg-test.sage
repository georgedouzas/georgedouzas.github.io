# name: set-sq-preimage-neg-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Maps.sq_preimage_neg
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['-1', '(by norm_num)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (y : ℝ) (hy : y < 0), (fun x : ℝ => x ^ 2) ⁻¹' {y} = ∅ := by\n  intro y hy\n  ext x\n  simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]\n  intro h; nlinarith [sq_nonneg x]\nexact @key (-1) ((by norm_num))"),
    (['-1/4', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (y : ℝ) (hy : y < 0), (fun x : ℝ => x ^ 2) ⁻¹' {y} = ∅ := by\n  intro y hy\n  ext x\n  simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]\n  intro h; nlinarith [sq_nonneg x]\nexact @key (-1/4) ((by norm_num))"),
    (['-9', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (y : ℝ) (hy : y < 0), (fun x : ℝ => x ^ 2) ⁻¹' {y} = ∅ := by\n  intro y hy\n  ext x\n  simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_empty_iff_false, iff_false]\n  intro h; nlinarith [sq_nonneg x]\nexact @key (-9) ((by norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
