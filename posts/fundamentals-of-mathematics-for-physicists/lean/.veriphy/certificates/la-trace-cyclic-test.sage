# name: la-trace-cyclic-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Contraction.trace_cyclic file "la-trace-cyclic-test.sage"
# description: tr (X Y) = tr (Y X) at n = 0 and explicit 2 × 2 and 3 × 3 matrices, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0'], True, 'simp [Matrix.trace]'),
    (['2', '!![1, 2; 3, 4]', '!![0, 1; -1, 2]'], False),
    (['3', '!![1, 2, 0; 0, 3, 1; 0, 0, 5]', '!![1, 0, 0; 1, 1, 0; 0, 2, 1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
