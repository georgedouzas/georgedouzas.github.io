# name: ca-holo-rational-zero-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.rational_eq_zero_iff
# description: (z - 1)/(z^2 + 1) = 0 iff z = 1, at z = 1, 0, 2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '(by norm_num)'], 'have h : ((1 : ℂ)) ^ 2 + 1 ≠ 0 := by norm_num [Complex.ext_iff, pow_two]\nrw [div_eq_zero_iff, sub_eq_zero]\ntauto', True),
    (['0', '(by norm_num)'], 'have h : ((0 : ℂ)) ^ 2 + 1 ≠ 0 := by norm_num [Complex.ext_iff, pow_two]\nrw [div_eq_zero_iff, sub_eq_zero]\ntauto', False),
    (['2', '(by norm_num)'], 'have h : ((2 : ℂ)) ^ 2 + 1 ≠ 0 := by norm_num [Complex.ext_iff, pow_two]\nrw [div_eq_zero_iff, sub_eq_zero]\ntauto', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
