# name: fa-normalization-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.FourierAnalysis.SphericalHarmonics.normalization file "fa-normalization-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
PROOF = 'have hπ : Real.pi ≠ 0 := Real.pi_ne_zero\nfield_simp\nnorm_num [Nat.factorial]'
CASES = [(['0', '0', 'le_refl 0'], True), (['2', '1', '(by norm_num)'], False), (['3', '3', 'le_refl 3'], False), (['3', '0', '(by norm_num)'], False)]
cases = CASES if CASES is not None else default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
