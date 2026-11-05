# name: ra-mvt-sin-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.mvt_sin file "ra-mvt-sin-test.sage"
# description: cos(π/2) = 0 = (sin kπ - sin 0)/kπ and π/2 ∈ (0, kπ)
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨by rw [Real.cos_pi_div_two, Real.sin_nat_mul_pi, Real.sin_zero]; simp, ⟨by positivity, ?_⟩⟩\nnorm_num\nnlinarith [Real.pi_pos]'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['5', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
