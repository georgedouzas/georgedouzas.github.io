# name: ra-ssup-ico-not-mem-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sSup_Ico_not_mem file "ra-ssup-ico-not-mem-test.sage"
# description: sup [a, b) = b, not attained, at [0, 1) and other intervals
SEED = 0  # the instances are listed, no sampling
PROOF = 'exact ⟨isLUB_Ico (by norm_num), csSup_Ico (by norm_num), by norm_num⟩'
CASES = [(['0', '1', '(by norm_num)'], True, None), (['-2', '3', '(by norm_num)'], False, None), (['1 / 2', '3 / 4', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
