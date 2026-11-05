# name: set-class-square-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.class_square
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), {y : ℝ | y ^ 2 = x ^ 2} = {x, -x} ∧ (x ≠ 0 → x ≠ -x) := by\n  intro x\n  refine ⟨?_, fun hx h => hx (by linarith)⟩\n  ext y\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  constructor\n  · intro h\n    have : (y - x) * (y + x) = 0 := by ring_nf; linarith\n    rcases mul_eq_zero.1 this with h1 | h1\n    · left; linarith\n    · right; linarith\n  · rintro (rfl | rfl) <;> ring\nexact @key (0)'),
    (['2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), {y : ℝ | y ^ 2 = x ^ 2} = {x, -x} ∧ (x ≠ 0 → x ≠ -x) := by\n  intro x\n  refine ⟨?_, fun hx h => hx (by linarith)⟩\n  ext y\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  constructor\n  · intro h\n    have : (y - x) * (y + x) = 0 := by ring_nf; linarith\n    rcases mul_eq_zero.1 this with h1 | h1\n    · left; linarith\n    · right; linarith\n  · rintro (rfl | rfl) <;> ring\nexact @key (2)'),
    (['-3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), {y : ℝ | y ^ 2 = x ^ 2} = {x, -x} ∧ (x ≠ 0 → x ≠ -x) := by\n  intro x\n  refine ⟨?_, fun hx h => hx (by linarith)⟩\n  ext y\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  constructor\n  · intro h\n    have : (y - x) * (y + x) = 0 := by ring_nf; linarith\n    rcases mul_eq_zero.1 this with h1 | h1\n    · left; linarith\n    · right; linarith\n  · rintro (rfl | rfl) <;> ring\nexact @key (-3)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
