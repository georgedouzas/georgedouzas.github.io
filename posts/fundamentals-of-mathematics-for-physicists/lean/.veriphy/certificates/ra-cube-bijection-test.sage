# name: ra-cube-bijection-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.cube_bijection file "ra-cube-bijection-test.sage"
# description: the cube root of y found by Sage and tan' = 1 + tan² at x = 0
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['0', '0', '(by norm_num)'], True, 'refine ⟨⟨0, by norm_num, fun z hz => by simpa using hz⟩, ?_⟩\nrw [Real.deriv_tan, Real.tan_zero]; norm_num'), (['8', '0', '(by norm_num)'], False, 'refine ⟨⟨2, by norm_num, fun z hz => ?_⟩, ?_⟩\n· nlinarith [sq_nonneg (z - 2), sq_nonneg (z + 1), sq_nonneg z]\n· rw [Real.deriv_tan, Real.tan_zero]; norm_num'), (['-27', '0', '(by norm_num)'], False, 'refine ⟨⟨-3, by norm_num, fun z hz => ?_⟩, ?_⟩\n· nlinarith [sq_nonneg (z + 3), sq_nonneg (z - 1), sq_nonneg z]\n· rw [Real.deriv_tan, Real.tan_zero]; norm_num')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
