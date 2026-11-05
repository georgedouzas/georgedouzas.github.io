# name: la-alternating-repeated-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.alternating_repeated file "la-alternating-repeated-test.sage"
# description: an alternating map vanishes on repeated arguments: the zero map, and the determinant on families of ℝ² and ℝ³ with two equal rows, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['(Fin 2)', '(Fin 2 → ℝ)', 'ℝ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', '0', '![![1, 2], ![1, 2]]', '0', '1', 'rfl', '(by decide)'], True, 'rfl'),
    (['(Fin 2)', '(Fin 2 → ℝ)', 'ℝ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Matrix.detRowAlternating', '![![1, 2], ![1, 2]]', '0', '1', 'rfl', '(by decide)'], False, 'show Matrix.det !![(1 : ℝ), 2; 1, 2] = 0\nnorm_num [Matrix.det_fin_two_of]'),
    (['(Fin 3)', '(Fin 3 → ℝ)', 'ℝ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Matrix.detRowAlternating', '![![1, 2, 3], ![0, 1, 4], ![1, 2, 3]]', '0', '2', 'rfl', '(by decide)'], False, 'show Matrix.det !![(1 : ℝ), 2, 3; 0, 1, 4; 1, 2, 3] = 0\nnorm_num [Matrix.det_fin_three]'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
