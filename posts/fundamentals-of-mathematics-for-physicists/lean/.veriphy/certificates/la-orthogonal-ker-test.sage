# name: la-orthogonal-ker-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearMaps.orthogonal_ker_trivial file "la-orthogonal-ker-test.sage"
# description: an orthogonal matrix has trivial kernel, at n = 0 and at rotations of ℝ² and ℝ³ (the only kernel vector, 0).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rfl'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '(by ext i j; exact i.elim0)', '![]', '(by ext i; exact i.elim0)'], True, 'ext i\nexact i.elim0'),
    (['2', '!![0, -1; 1, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])', '0', '(by simp)'], False),
    (['3', '!![1, 0, 0; 0, 0, -1; 0, 1, 0]', '(by ext i j; fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ])', '0', '(by simp)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
