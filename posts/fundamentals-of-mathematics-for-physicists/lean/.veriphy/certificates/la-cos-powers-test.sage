# name: la-cos-powers-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Dimension.cos_powers_linearIndependent file "la-cos-powers-test.sage"
# description: 1, cos φ, …, cosᴺ φ are independent at N = 0, 1, 2: their values at 0, π/2, π form a non-singular matrix.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True, 'refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin 1 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0] k)) ?_\nexact Veriphy.Linear.linearIndependent_of_minor _ ![0] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi, Real.sin_pi])'),
    (['1'], False, 'refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin 2 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0, Real.pi / 2] k)) ?_\nexact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi, Real.sin_pi])'),
    (['2'], False, 'refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin 3 => LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (![0, Real.pi / 2, Real.pi] k)) ?_\nexact Veriphy.Linear.linearIndependent_of_minor _ ![0, 1, 2] (by norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, Fin.succAbove_of_le_castSucc, LinearMap.pi_apply, LinearMap.proj_apply, Function.comp_apply, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.cos_pi, Real.sin_pi])'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
