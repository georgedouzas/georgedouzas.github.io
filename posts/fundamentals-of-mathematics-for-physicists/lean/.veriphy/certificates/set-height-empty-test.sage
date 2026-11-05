# name: set-height-empty-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Maps.height_preimage_empty
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2', '3', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (ha : 1 < a), sphere ∩ height ⁻¹' Set.Icc a b = ∅ := by\n  intro a b ha\n  ext p\n  simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc,\n    Set.mem_empty_iff_false, iff_false, not_and]\n  intro h h1 _\n  nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]\nexact @key (2) (3) ((by norm_num))"),
    (['3/2', '1', '(by norm_num)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (ha : 1 < a), sphere ∩ height ⁻¹' Set.Icc a b = ∅ := by\n  intro a b ha\n  ext p\n  simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc,\n    Set.mem_empty_iff_false, iff_false, not_and]\n  intro h h1 _\n  nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]\nexact @key (3/2) (1) ((by norm_num))"),
    (['5', '10', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℝ) (ha : 1 < a), sphere ∩ height ⁻¹' Set.Icc a b = ∅ := by\n  intro a b ha\n  ext p\n  simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Icc,\n    Set.mem_empty_iff_false, iff_false, not_and]\n  intro h h1 _\n  nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]\nexact @key (5) (10) ((by norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
