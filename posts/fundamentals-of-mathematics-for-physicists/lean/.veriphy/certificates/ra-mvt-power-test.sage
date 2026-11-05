# name: ra-mvt-power-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.mvt_power file "ra-mvt-power-test.sage"
# description: for x³ on [0, b]: 3(b/√3)² = b², 0 < b/√3 < b, and 0.577 b < b/√3 < 0.578 b
SEED = 0  # the instances are listed, no sampling
PROOF = "have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)\nhave hs : 0 < Real.sqrt 3 := by positivity\nhave hs1 : 1.732 < Real.sqrt 3 := by rw [Real.lt_sqrt (by norm_num)]; norm_num\nhave hs2 : Real.sqrt 3 < 1.7321 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num\nrefine ⟨?_, by positivity, ?_, ?_, ?_⟩\n· field_simp; rw [h3]; ring\n· rw [div_lt_iff₀ hs]; nlinarith\n· rw [lt_div_iff₀ hs]; nlinarith\n· rw [div_lt_iff₀ hs]; nlinarith"
CASES = [(['2', '(by norm_num)'], False, None), (['1', '(by norm_num)'], True, None), (['3', '(by norm_num)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
