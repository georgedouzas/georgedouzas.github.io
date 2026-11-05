# name: la-triple-product-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.triple_product_det file "la-triple-product-test.sage"
# description: u · (v × w) = det(u | v | w) at the zero vectors, the standard basis and explicit vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0'], True, 'rw [show Matrix.det ![(0 : Fin 3 → ℝ), 0, 0] = Matrix.det (Matrix.of ![(0 : Fin 3 → ℝ), 0, 0]) from rfl,\n  Matrix.det_fin_three]\nsimp [cross_apply, dotProduct, Fin.sum_univ_succ]\n<;> ring'),
    (['![1, 0, 0]', '![0, 1, 0]', '![0, 0, 1]'], False, 'rw [show Matrix.det ![(![1, 0, 0] : Fin 3 → ℝ), ![0, 1, 0], ![0, 0, 1]] = Matrix.det (Matrix.of ![(![1, 0, 0] : Fin 3 → ℝ), ![0, 1, 0], ![0, 0, 1]]) from rfl,\n  Matrix.det_fin_three]\nsimp [cross_apply, dotProduct, Fin.sum_univ_succ]\n<;> ring'),
    (['![1, 2, 3]', '![0, 1, -1]', '![2, 0, 1]'], False, 'rw [show Matrix.det ![(![1, 2, 3] : Fin 3 → ℝ), ![0, 1, -1], ![2, 0, 1]] = Matrix.det (Matrix.of ![(![1, 2, 3] : Fin 3 → ℝ), ![0, 1, -1], ![2, 0, 1]]) from rfl,\n  Matrix.det_fin_three]\nsimp [cross_apply, dotProduct, Fin.sum_univ_succ]\n<;> ring'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
