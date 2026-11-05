# name: la-polarisation-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.SymmetricForms.polarisation file "la-polarisation-test.sage"
# description: 2 B(v, w) = B(v + w, v + w) − B(v, v) − B(w, w) for symmetric B, at explicit B, v, w, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = "norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]"  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by simp)', '![]', '![]'], True, 'simp'),
    (['2', '!![1, 2; 2, 4]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '![1, 3]', '![2, -1]'], False),
    (['3', '!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num)', '![1, 2, -1]', '![0, 1, 1 / 2]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
