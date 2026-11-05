# name: la-antisym-of-diag-zero-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.antisym_of_diag_zero file "la-antisym-of-diag-zero-test.sage"
# description: a form vanishing on the diagonal is antisymmetric, at the zero form and the determinant form of ℝ², by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = "norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]"  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', '0', '(fun _ => rfl)', '![1, 2]', '![3, -1]'], True, 'simp'),
    (['(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', "Matrix.toBilin' !![0, 1; -1, 0]", "(by intro v; simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]; ring)", '![1, 2]', '![3, -1]'], False),
    (['(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', "Matrix.toBilin' !![0, 3; -3, 0]", "(by intro v; simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ]; ring)", '![1 / 2, 0]', '![2, 5]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
