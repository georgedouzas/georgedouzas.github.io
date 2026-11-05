# name: ra-one-sub-inv-sup-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.one_sub_inv_sup file "ra-one-sub-inv-sup-test.sage"
# description: sup {1 - 1/n} = 1 in the ε form, at ε = 1, 1/10, 1/1000
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨fun n hn => ?_, fun n hn => ?_⟩\n· have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn\n  have : 0 < 1 / (n : ℝ) := by positivity\n  linarith\n· have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn\n  rw [sub_lt_sub_iff_left, div_lt_iff₀ hn0]\n  rw [div_lt_iff₀ (by norm_num)] at hn\n  linarith'
CASES = [(['1', '(by norm_num)'], True, None), (['1 / 10', '(by norm_num)'], False, None), (['1 / 1000', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
