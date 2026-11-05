# name: la-finrank-tensor-general-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.TensorProducts.finrank_tensor_general file "la-finrank-tensor-general-test.sage"
# description: dim (V ⊗ W) = dim V · dim W at V = ℚ⁰, ℚ², ℚ³ and W = ℚ³, ℚ, ℚ⁴, by the finrank formulas.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Module.finrank_tensorProduct]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 0 → ℚ)', '(Fin 3 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], True),
    (['ℚ', '(Fin 2 → ℚ)', 'ℚ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
    (['ℚ', '(Fin 3 → ℚ)', '(Fin 4 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
