# name: fa-odd-tail-bound-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.FourierAnalysis.SeriesComputations.odd_tail_bound file "fa-odd-tail-bound-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
PROOF = 'norm_num [Finset.sum_range_succ, KaluzaKlein.FourierAnalysis.SeriesComputations.oddInvSq]'
CASES = [(['1', '0', 'le_refl 1'], True), (['1', '3', 'le_refl 1'], False), (['2', '4', '(by norm_num)'], False), (['5', '2', '(by norm_num)'], False)]
cases = CASES if CASES is not None else default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
