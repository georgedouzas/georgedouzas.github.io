# name: la-antisym-add-smul-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.antisym_add_smul file "la-antisym-add-smul-test.sage"
# description: A + c B is antisymmetric for antisymmetric A, B, at n = 0, 2, 3 with explicit matrices, entry by entry.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext i j\nfin_cases i <;> fin_cases j <;> norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '0', '(by ext i j; exact i.elim0)', '(by ext i j; exact i.elim0)'], True, 'ext i j\nexact i.elim0'),
    (['2', '!![0, 1; -1, 0]', '!![0, -3; 3, 0]', '2', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)'], False),
    (['3', '!![0, -5, 2; 5, 0, -7; -2, 7, 0]', '!![0, 0, 0; 0, 0, -1; 0, 1, 0]', '-1 / 2', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
