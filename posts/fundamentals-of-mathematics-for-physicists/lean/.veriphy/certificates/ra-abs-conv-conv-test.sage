# name: ra-abs-conv-conv-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.abs_conv_conv file "ra-abs-conv-conv-test.sage"
# description: absolute convergence gives convergence, at sequences whose |·|-summability is given
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['(fun _ : ℕ => (0 : ℝ))', '(summable_zero.congr (fun k => by simp))'], True, 'exact summable_zero'), (['(fun k : ℕ => (1 / 2 : ℝ) ^ k)', '(summable_geometric_two.congr (fun k => (abs_of_nonneg (by positivity)).symm))'], False, 'exact summable_geometric_two'), (['(fun k : ℕ => (1 / 3 : ℝ) ^ k)', '((summable_geometric_of_lt_one (by norm_num) (by norm_num)).congr (fun k => (abs_of_nonneg (by positivity)).symm))'], False, 'exact summable_geometric_of_lt_one (by norm_num) (by norm_num)')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
