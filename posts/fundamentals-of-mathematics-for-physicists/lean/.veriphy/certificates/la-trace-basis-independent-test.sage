# name: la-trace-basis-independent-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.Linear.trace_basis_independent file "la-trace-basis-independent-test.sage"
# description: tr (Q T P) = tr T for Q P = 1, at n = 0 and at explicit 2 × 2 and 3 × 3 matrices, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '0', '(by ext i j; exact i.elim0)'], True, 'simp [Matrix.trace]'),
    (['2', '!![1, 2; 3, 4]', '!![2, 1; 1, 1]', '!![1, -1; -1, 2]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])'], False),
    (['3', '!![1, 2, 0; 0, 3, 1; 0, 0, 5]', '!![1, 0, 0; 1, 1, 0; 0, 0, 1]', '!![1, 0, 0; -1, 1, 0; 0, 0, 1]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
