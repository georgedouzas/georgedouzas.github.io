# name: ra-isgreatest-three-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.isGreatest_three file "ra-isgreatest-three-test.sage"
# description: the maximum of a three-element set, {1, 4, 2} among them
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨⟨by simp, fun x hx => ?_⟩, IsGreatest.csSup_eq ⟨by simp, fun x hx => ?_⟩⟩ <;> (simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num)'
CASES = [(['1', '4', '2', '(by norm_num)', '(by norm_num)'], False, None), (['0', '0', '0', '(by norm_num)', '(by norm_num)'], True, None), (['-1', '3 / 2', '2 / 3', '(by norm_num)', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
