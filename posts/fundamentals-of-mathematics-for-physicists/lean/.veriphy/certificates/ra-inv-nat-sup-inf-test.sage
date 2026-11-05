# name: ra-inv-nat-sup-inf-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.inv_nat_sup_inf file "ra-inv-nat-sup-inf-test.sage"
# description: {1/n}: max 1 at n = 1, inf 0 not attained; ε form at ε = 1, 1/10, 1/1000
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨fun n hn => ?_, by norm_num, fun n hn => ?_⟩\n· have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn\n  exact ⟨by positivity, by rw [div_le_one (by positivity)]; exact h1⟩\n· have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn\n  rw [div_lt_iff₀ hn0]\n  rw [div_lt_iff₀ (by norm_num)] at hn\n  linarith'
CASES = [(['1', '(by norm_num)'], True, None), (['1 / 10', '(by norm_num)'], False, None), (['1 / 1000', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
