# name: ra-completeness-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.completeness file "ra-completeness-test.sage"
# description: a least upper bound exists: Sage computes the supremum of each set, Lean checks it is the least upper bound
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['({1, 4, 2} : Set ℝ)', '⟨1, by simp⟩', '⟨4, fun x hx => by simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num⟩'], False, 'exact ⟨4, fun x hx => by simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num, fun b hb => hb (by simp)⟩'), (['Set.Icc (0 : ℝ) 1', '⟨0, by norm_num⟩', 'bddAbove_Icc'], False, 'exact ⟨1, isLUB_Icc (by norm_num)⟩'), (['({0} : Set ℝ)', '⟨0, rfl⟩', 'bddAbove_singleton'], True, 'exact ⟨0, isLUB_singleton⟩'), (['Set.Ico (-1 : ℝ) 2', '⟨0, by norm_num⟩', 'bddAbove_Ico'], False, 'exact ⟨2, isLUB_Ico (by norm_num)⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
