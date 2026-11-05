# name: ra-sinf-eq-neg-ssup-neg-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sInf_eq_neg_sSup_neg file "ra-sinf-eq-neg-ssup-neg-test.sage"
# description: inf S = -sup(-S) on intervals, both sides computed exactly
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['(∅ : Set ℝ)'], True, 'simp'), (['Set.Icc (0 : ℝ) 1'], False, 'rw [Set.neg_Icc, csInf_Icc (by norm_num), csSup_Icc (by norm_num)]; norm_num'), (['Set.Icc (-2 : ℝ) 3'], False, 'rw [Set.neg_Icc, csInf_Icc (by norm_num), csSup_Icc (by norm_num)]; norm_num'), (['Set.Icc (1 : ℝ) 1'], False, 'rw [Set.neg_Icc, csInf_Icc (by norm_num), csSup_Icc (by norm_num)]; norm_num')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
