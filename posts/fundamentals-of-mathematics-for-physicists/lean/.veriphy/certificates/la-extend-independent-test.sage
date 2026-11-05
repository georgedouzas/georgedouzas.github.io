# name: la-extend-independent-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearIndependence.extend_independent file "la-extend-independent-test.sage"
# description: an independent family extended by a vector outside its span, at n = 0, 1, 2 in ℚ² and ℚ³: the extended family has a non-singular minor (checked by evaluation).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 2 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', '0', '![]', '![1, 0]', 'linearIndependent_empty_type', '(by simp)'], True, 'exact Veriphy.Linear.linearIndependent_of_minor _ ![0] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])'),
    (['ℚ', '(Fin 2 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', '1', '![![1, 0]]', '![0, 1]', '(Veriphy.Linear.linearIndependent_of_minor _ ![0] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]))', '(by rw [Submodule.mem_span_range_iff_exists_fun]; rintro ⟨c, hc⟩; have h0 := congrFun hc 0; have h1 := congrFun hc 1; simp [Fin.sum_univ_succ] at h0 h1 <;> linarith)'], False, 'exact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])'),
    (['ℚ', '(Fin 3 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', '2', '![![1, 2, 0], ![0, 1, 1]]', '![0, 0, 1]', '(Veriphy.Linear.linearIndependent_of_minor _ ![0, 1] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc]))', '(by rw [Submodule.mem_span_range_iff_exists_fun]; rintro ⟨c, hc⟩; have h0 := congrFun hc 0; have h1 := congrFun hc 1; have h2 := congrFun hc 2; simp [Fin.sum_univ_succ] at h0 h1 h2 <;> linarith)'], False, 'exact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc])'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
