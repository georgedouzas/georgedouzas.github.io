# name: set-range-sq-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Maps.range_sq_add
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ), Set.range (fun x : ℝ => x ^ 2 + c) = Set.Ici c := by\n  intro c\n  ext y\n  simp only [Set.mem_range, Set.mem_Ici]\n  constructor\n  · rintro ⟨x, rfl⟩; nlinarith [sq_nonneg x]\n  · intro h\n    exact ⟨Real.sqrt (y - c), by rw [Real.sq_sqrt (by linarith)]; ring⟩\nexact @key (0)'),
    (['1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ), Set.range (fun x : ℝ => x ^ 2 + c) = Set.Ici c := by\n  intro c\n  ext y\n  simp only [Set.mem_range, Set.mem_Ici]\n  constructor\n  · rintro ⟨x, rfl⟩; nlinarith [sq_nonneg x]\n  · intro h\n    exact ⟨Real.sqrt (y - c), by rw [Real.sq_sqrt (by linarith)]; ring⟩\nexact @key (1)'),
    (['-3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ), Set.range (fun x : ℝ => x ^ 2 + c) = Set.Ici c := by\n  intro c\n  ext y\n  simp only [Set.mem_range, Set.mem_Ici]\n  constructor\n  · rintro ⟨x, rfl⟩; nlinarith [sq_nonneg x]\n  · intro h\n    exact ⟨Real.sqrt (y - c), by rw [Real.sq_sqrt (by linarith)]; ring⟩\nexact @key (-3)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
