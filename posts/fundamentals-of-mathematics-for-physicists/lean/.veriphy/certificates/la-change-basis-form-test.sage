# name: la-change-basis-form-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.BilinearForms.change_basis_form file "la-change-basis-form-test.sage"
# description: B(A x, A y) = xᵀ (Aᵀ B A) y at n = 0 and explicit A, B, x, y in two and three dimensions, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [Matrix.mulVec, dotProduct, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '![]', '![]'], True, 'simp'),
    (['2', '!![2, 1; 1, 2]', '!![1, 0; 0, -1]', '![1, 2]', '![3, -1]'], False),
    (['3', '!![1, 0, 1; 1, 1, 0; 0, 1, 1]', '!![1, 2, 0; 2, 0, 1; 0, 1, -1]', '![1, 0, -1]', '![2, 1, 1 / 2]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
