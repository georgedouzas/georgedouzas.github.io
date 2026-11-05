# name: ca-holo-mobius-ne-zero-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.mobius_deriv_ne_zero
# description: 2i/(z + i)^2 ≠ 0 at z = 0, 1, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by norm_num [Complex.ext_iff])'], 'exact div_ne_zero (mul_ne_zero two_ne_zero Complex.I_ne_zero) (pow_ne_zero 2 (by norm_num [Complex.ext_iff]))', True),
    (['1', '(by norm_num [Complex.ext_iff])'], 'exact div_ne_zero (mul_ne_zero two_ne_zero Complex.I_ne_zero) (pow_ne_zero 2 (by norm_num [Complex.ext_iff]))', False),
    (['Complex.I', '(by norm_num [Complex.ext_iff])'], 'exact div_ne_zero (mul_ne_zero two_ne_zero Complex.I_ne_zero) (pow_ne_zero 2 (by norm_num [Complex.ext_iff]))', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
