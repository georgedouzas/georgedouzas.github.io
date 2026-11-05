# name: ra-inv-unbounded-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.inv_unbounded file "ra-inv-unbounded-test.sage"
# description: a point x ∈ (0, 1] with 1/x > M, chosen by Sage
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['0'], True, 'exact ⟨1 / 2, by norm_num, by norm_num⟩'), (['10'], False, 'exact ⟨1 / 11, by norm_num, by norm_num⟩'), (['1000'], False, 'exact ⟨1 / 1001, by norm_num, by norm_num⟩'), (['-5'], False, 'exact ⟨1, by norm_num, by norm_num⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
