# name: ca-holo-deriv-rational-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexDerivative.deriv_rational
# description: the quotient rule for (z - 1)/(z^2 + 1) at 0, 1, 2i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by norm_num [Complex.ext_iff, pow_two])'], "have hz := (by norm_num [Complex.ext_iff, pow_two] : (0 : ℂ) ^ 2 + 1 ≠ 0)\nshow deriv (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) (0 : ℂ) = (-(0 : ℂ) ^ 2 + 2 * (0 : ℂ) + 1) / ((0 : ℂ) ^ 2 + 1) ^ 2\nhave h := ((hasDerivAt_id' (0 : ℂ)).sub_const 1).div ((hasDerivAt_pow 2 (0 : ℂ)).add_const 1) hz\nrefine h.deriv.trans ?_\nsimp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one, Nat.cast_ofNat]\nring", True),
    (['1', '(by norm_num [Complex.ext_iff, pow_two])'], "have hz := (by norm_num [Complex.ext_iff, pow_two] : (1 : ℂ) ^ 2 + 1 ≠ 0)\nshow deriv (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) (1 : ℂ) = (-(1 : ℂ) ^ 2 + 2 * (1 : ℂ) + 1) / ((1 : ℂ) ^ 2 + 1) ^ 2\nhave h := ((hasDerivAt_id' (1 : ℂ)).sub_const 1).div ((hasDerivAt_pow 2 (1 : ℂ)).add_const 1) hz\nrefine h.deriv.trans ?_\nsimp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one, Nat.cast_ofNat]\nring", False),
    (['(2 * Complex.I)', '(by norm_num [Complex.ext_iff, pow_two])'], "have hz := (by norm_num [Complex.ext_iff, pow_two] : (2 * Complex.I : ℂ) ^ 2 + 1 ≠ 0)\nshow deriv (fun w : ℂ => (w - 1) / (w ^ 2 + 1)) (2 * Complex.I) = (-(2 * Complex.I) ^ 2 + 2 * (2 * Complex.I) + 1) / ((2 * Complex.I) ^ 2 + 1) ^ 2\nhave h := ((hasDerivAt_id' (2 * Complex.I)).sub_const 1).div ((hasDerivAt_pow 2 (2 * Complex.I)).add_const 1) hz\nrefine h.deriv.trans ?_\nsimp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one, Nat.cast_ofNat]\nring", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
