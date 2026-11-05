# name: ca-holo-exp-strip-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.exp_strip_onto
# description: w = i, 1 + i, -1 + 2i are images of the strip; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['Complex.I'], "constructor\n· rintro ⟨z, h1, h2, hz⟩\n  rw [← hz, Complex.exp_im]\n  exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi h1 h2)\n· intro hw\n  have hw0 : (Complex.I) ≠ 0 := Complex.I_ne_zero\n  refine ⟨Complex.log (Complex.I), ?_, ?_, Complex.exp_log hw0⟩\n  · rw [Complex.log_im]\n    exact lt_of_le_of_ne (Complex.arg_nonneg_iff.mpr hw.le) fun h => hw.ne' (Complex.arg_eq_zero_iff.mp h.symm).2\n  · rw [Complex.log_im]\n    exact lt_of_le_of_ne (Complex.arg_le_pi _) fun h => by linarith [(Complex.arg_eq_pi_iff.mp h).2]", True),
    (['(1 + Complex.I)'], "constructor\n· rintro ⟨z, h1, h2, hz⟩\n  rw [← hz, Complex.exp_im]\n  exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi h1 h2)\n· intro hw\n  have hw0 : (1 + Complex.I) ≠ 0 := by norm_num [Complex.ext_iff]\n  refine ⟨Complex.log (1 + Complex.I), ?_, ?_, Complex.exp_log hw0⟩\n  · rw [Complex.log_im]\n    exact lt_of_le_of_ne (Complex.arg_nonneg_iff.mpr hw.le) fun h => hw.ne' (Complex.arg_eq_zero_iff.mp h.symm).2\n  · rw [Complex.log_im]\n    exact lt_of_le_of_ne (Complex.arg_le_pi _) fun h => by linarith [(Complex.arg_eq_pi_iff.mp h).2]", False),
    (['(-1 + 2 * Complex.I)'], "constructor\n· rintro ⟨z, h1, h2, hz⟩\n  rw [← hz, Complex.exp_im]\n  exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi h1 h2)\n· intro hw\n  have hw0 : (-1 + 2 * Complex.I) ≠ 0 := by norm_num [Complex.ext_iff]\n  refine ⟨Complex.log (-1 + 2 * Complex.I), ?_, ?_, Complex.exp_log hw0⟩\n  · rw [Complex.log_im]\n    exact lt_of_le_of_ne (Complex.arg_nonneg_iff.mpr hw.le) fun h => hw.ne' (Complex.arg_eq_zero_iff.mp h.symm).2\n  · rw [Complex.log_im]\n    exact lt_of_le_of_ne (Complex.arg_le_pi _) fun h => by linarith [(Complex.arg_eq_pi_iff.mp h).2]", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
