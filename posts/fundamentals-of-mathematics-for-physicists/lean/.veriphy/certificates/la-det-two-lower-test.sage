# name: la-det-two-lower-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Components.det_two_lower file "la-det-two-lower-test.sage"
# description: det (Aᵀ B A) = (det A)² det B at n = 0 and explicit 2 × 2 and 3 × 3 matrices, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Matrix.det_fin_two, Matrix.det_fin_three, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]\n<;> norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0'], True, 'simp'),
    (['2', '!![2, 1; 1, 2]', '!![1, 0; 0, -1]'], False),
    (['3', '!![1, 0, 1; 1, 1, 0; 0, 1, 1]', '!![1, 2, 0; 2, 0, 1; 0, 1, -1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
