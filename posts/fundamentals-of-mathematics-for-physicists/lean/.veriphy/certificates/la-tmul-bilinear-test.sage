# name: la-tmul-bilinear-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.ElementaryTensors.tmul_bilinear file "la-tmul-bilinear-test.sage"
# description: (v + t v') ⊗ w = v ⊗ w + t (v' ⊗ w) in ℝ² ⊗ ℝ² and ℝ ⊗ ℝ³ at explicit vectors and scalars.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = "rw [add_tmul, smul_tmul']"  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['(Fin 2 → ℝ)', '(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', '0', '0', '0', '0'], True),
    (['(Fin 2 → ℝ)', '(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', '![1, 2]', '![0, 1]', '![3, -1]', '2'], False),
    (['ℝ', '(Fin 3 → ℝ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', '1', '-2', '![1, 0, 1]', '1 / 2'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
