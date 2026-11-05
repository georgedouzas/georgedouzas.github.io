# name: set-upper-part-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Subspace.upper_part_open
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (t : ℝ), IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t < p.2}) := by\n  intro t\n  exact ((isOpen_lt continuous_const continuous_snd).preimage continuous_subtype_val)\nexact @key (0)"),
    (['1/2'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (t : ℝ), IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t < p.2}) := by\n  intro t\n  exact ((isOpen_lt continuous_const continuous_snd).preimage continuous_subtype_val)\nexact @key (1/2)"),
    (['-1'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (t : ℝ), IsOpen ((Subtype.val : circle → ℝ × ℝ) ⁻¹' {p | t < p.2}) := by\n  intro t\n  exact ((isOpen_lt continuous_const continuous_snd).preimage continuous_subtype_val)\nexact @key (-1)"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
