# name: ra-comparison-test-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.comparison_test file "ra-comparison-test-test.sage"
# description: the comparison test at explicit dominations by geometric series
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['(fun _ : ℕ => (0 : ℝ))', '(fun k : ℕ => (1 / 2 : ℝ) ^ k)', '(fun _ => le_rfl)', '(fun k => by positivity)', 'summable_geometric_two'], True, 'exact summable_zero'), (['(fun k : ℕ => (1 / 3 : ℝ) ^ k)', '(fun k : ℕ => (1 / 2 : ℝ) ^ k)', '(fun k => by positivity)', '(fun k => pow_le_pow_left₀ (by norm_num) (by norm_num) k)', 'summable_geometric_two'], False, 'exact summable_geometric_of_lt_one (by norm_num) (by norm_num)'), (['(fun k : ℕ => (1 / 4 : ℝ) ^ k)', '(fun k : ℕ => (1 / 2 : ℝ) ^ k)', '(fun k => by positivity)', '(fun k => pow_le_pow_left₀ (by norm_num) (by norm_num) k)', 'summable_geometric_two'], False, 'exact summable_geometric_of_lt_one (by norm_num) (by norm_num)')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
