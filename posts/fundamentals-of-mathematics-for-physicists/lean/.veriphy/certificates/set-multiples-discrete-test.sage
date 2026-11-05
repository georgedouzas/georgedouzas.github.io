# name: set-multiples-discrete-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Subspace.multiples_discrete
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ) (hT : 0 < T), DiscreteTopology (range (fun k : ℤ => (k : ℝ) * T)) := by\n  intro T hT\n  refine DiscreteTopology.of_forall_le_dist hT ?_\n  rintro ⟨x, k, rfl⟩ ⟨y, l, rfl⟩ hxy\n  have hkl : k ≠ l := fun h => hxy (by subst h; rfl)\n  rw [Subtype.dist_eq, Real.dist_eq, ← sub_mul, abs_mul, abs_of_pos hT]\n  have : (1 : ℝ) ≤ |(k : ℝ) - l| := by\n    rw [← Int.cast_sub, ← Int.cast_abs]\n    exact_mod_cast Int.one_le_abs (sub_ne_zero.2 hkl)\n  nlinarith\nexact @key (1) ((by norm_num))'),
    (['2 * Real.pi', '(by positivity)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ) (hT : 0 < T), DiscreteTopology (range (fun k : ℤ => (k : ℝ) * T)) := by\n  intro T hT\n  refine DiscreteTopology.of_forall_le_dist hT ?_\n  rintro ⟨x, k, rfl⟩ ⟨y, l, rfl⟩ hxy\n  have hkl : k ≠ l := fun h => hxy (by subst h; rfl)\n  rw [Subtype.dist_eq, Real.dist_eq, ← sub_mul, abs_mul, abs_of_pos hT]\n  have : (1 : ℝ) ≤ |(k : ℝ) - l| := by\n    rw [← Int.cast_sub, ← Int.cast_abs]\n    exact_mod_cast Int.one_le_abs (sub_ne_zero.2 hkl)\n  nlinarith\nexact @key (2 * Real.pi) ((by positivity))'),
    (['1/2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T : ℝ) (hT : 0 < T), DiscreteTopology (range (fun k : ℤ => (k : ℝ) * T)) := by\n  intro T hT\n  refine DiscreteTopology.of_forall_le_dist hT ?_\n  rintro ⟨x, k, rfl⟩ ⟨y, l, rfl⟩ hxy\n  have hkl : k ≠ l := fun h => hxy (by subst h; rfl)\n  rw [Subtype.dist_eq, Real.dist_eq, ← sub_mul, abs_mul, abs_of_pos hT]\n  have : (1 : ℝ) ≤ |(k : ℝ) - l| := by\n    rw [← Int.cast_sub, ← Int.cast_abs]\n    exact_mod_cast Int.one_le_abs (sub_ne_zero.2 hkl)\n  nlinarith\nexact @key (1/2) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
