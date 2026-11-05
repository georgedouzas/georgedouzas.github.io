# name: ca-holo-inv-uy-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyRiemann.inv_uy
# description: the partial derivative uy of the parts of 1/z at (x, y) = (1, 0), (1, 2), (-1/2, 3); each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '0', '(by norm_num)'], "have h := (by norm_num : (1 : ℝ) ^ 2 + (0 : ℝ) ^ 2 ≠ 0)\nhave hd := (hasDerivAt_const (0 : ℝ) (1 : ℝ)).fun_div ((hasDerivAt_pow 2 (0 : ℝ)).const_add ((1 : ℝ) ^ 2)) h\nrefine hd.deriv.trans ?_\nrw [div_left_inj' (pow_ne_zero 2 h)]\nsimp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]\nring", True),
    (['1', '2', '(by norm_num)'], "have h := (by norm_num : (1 : ℝ) ^ 2 + (2 : ℝ) ^ 2 ≠ 0)\nhave hd := (hasDerivAt_const (2 : ℝ) (1 : ℝ)).fun_div ((hasDerivAt_pow 2 (2 : ℝ)).const_add ((1 : ℝ) ^ 2)) h\nrefine hd.deriv.trans ?_\nrw [div_left_inj' (pow_ne_zero 2 h)]\nsimp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]\nring", False),
    (['(-1 / 2)', '3', '(by norm_num)'], "have h := (by norm_num : (-1 / 2 : ℝ) ^ 2 + (3 : ℝ) ^ 2 ≠ 0)\nhave hd := (hasDerivAt_const (3 : ℝ) (-1 / 2 : ℝ)).fun_div ((hasDerivAt_pow 2 (3 : ℝ)).const_add ((-1 / 2 : ℝ) ^ 2)) h\nrefine hd.deriv.trans ?_\nrw [div_left_inj' (pow_ne_zero 2 h)]\nsimp only [Nat.cast_ofNat, show (2 : ℕ) - 1 = 1 from rfl, pow_one]\nring", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
