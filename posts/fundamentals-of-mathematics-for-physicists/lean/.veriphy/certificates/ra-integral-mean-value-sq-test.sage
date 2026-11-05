# name: ra-integral-mean-value-sq-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.integral_mean_value_sq file "ra-integral-mean-value-sq-test.sage"
# description: ∫₀ᵇ x² = (b/√3)² b and b/√3 ∈ [0, b], at b = 1 (c = 1/√3), 2, 3
SEED = 0  # the instances are listed, no sampling
PROOF = 'have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)\nhave hs : 1 ≤ Real.sqrt 3 := by rw [show (1 : ℝ) = Real.sqrt 1 by simp]; exact Real.sqrt_le_sqrt (by norm_num)\nrefine ⟨?_, by positivity, ?_⟩\n· rw [integral_pow, div_pow, h3]; norm_num\n· rw [div_le_iff₀ (by positivity)]; nlinarith'
CASES = [(['1', '(by norm_num)'], True, None), (['2', '(by norm_num)'], False, None), (['3', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
