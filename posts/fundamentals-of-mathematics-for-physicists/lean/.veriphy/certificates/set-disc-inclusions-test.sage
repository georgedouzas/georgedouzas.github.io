# name: set-disc-inclusions-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.disc_inclusions
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r : ℝ), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∩ {p | p.1 ^ 2 + p.2 ^ 2 < r} = ∅ := by\n  intro r\n  refine ⟨fun p (hp : _ < r) => (le_of_lt hp : _ ≤ r), fun p (hp : _ = r) => (le_of_eq hp : _ ≤ r), ?_⟩\n  ext p\n  simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]\n  intro h1 h2\n  linarith\nexact @key (1)'),
    (['0'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r : ℝ), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∩ {p | p.1 ^ 2 + p.2 ^ 2 < r} = ∅ := by\n  intro r\n  refine ⟨fun p (hp : _ < r) => (le_of_lt hp : _ ≤ r), fun p (hp : _ = r) => (le_of_eq hp : _ ≤ r), ?_⟩\n  ext p\n  simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]\n  intro h1 h2\n  linarith\nexact @key (0)'),
    (['4'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r : ℝ), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 < r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ⊆ {p | p.1 ^ 2 + p.2 ^ 2 ≤ r} ∧ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r} ∩ {p | p.1 ^ 2 + p.2 ^ 2 < r} = ∅ := by\n  intro r\n  refine ⟨fun p (hp : _ < r) => (le_of_lt hp : _ ≤ r), fun p (hp : _ = r) => (le_of_eq hp : _ ≤ r), ?_⟩\n  ext p\n  simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]\n  intro h1 h2\n  linarith\nexact @key (4)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
