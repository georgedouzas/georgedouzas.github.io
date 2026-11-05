# name: la-antisym-eq-combination-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearCombinations.antisym_eq_combination file "la-antisym-eq-combination-test.sage"
# description: X = X₃₂ L₁ + X₁₃ L₂ + X₂₁ L₃ for explicit antisymmetric X (zero, the exercise's matrix, L₁), entry by entry.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext i j\nfin_cases i <;> fin_cases j <;> norm_num [L1, L2, L3]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '(by simp)'], True),
    (['!![0, -5, 2; 5, 0, -7; -2, 7, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)'], False),
    (['!![0, 0, 0; 0, 0, -1; 0, 1, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)'], False),
    (['!![0, 1, -3; -1, 0, 1 / 2; 3, -1 / 2, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
