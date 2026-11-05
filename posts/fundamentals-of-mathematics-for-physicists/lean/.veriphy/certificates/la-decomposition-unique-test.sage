# name: la-decomposition-unique-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.SymmetricForms.decomposition_unique file "la-decomposition-unique-test.sage"
# description: the symmetric part of S + A is S, at n = 0 and explicit symmetric S and antisymmetric A (the hypothesis forces S' = S).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rfl'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '0', '0', '(by simp)', '(by simp)', '(by simp)', '(by simp)', 'rfl'], True),
    (['2', '!![1, 1 / 2; 1 / 2, 0]', '!![0, 3 / 2; -3 / 2, 0]', '!![1, 1 / 2; 1 / 2, 0]', '!![0, 3 / 2; -3 / 2, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', 'rfl'], False),
    (['3', '!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]', '!![0, -5, 2; 5, 0, -7; -2, 7, 0]', '!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]', '!![0, -5, 2; 5, 0, -7; -2, 7, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', 'rfl'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
