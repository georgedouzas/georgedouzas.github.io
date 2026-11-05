# name: la-finrank-dual-tensor-general-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.TensorResults.finrank_dual_tensor_general file "la-finrank-dual-tensor-general-test.sage"
# description: dim (V* ⊗ V) = (dim V)² at V = ℚ⁰, ℚ, ℚ² and ℚ³, by the finrank formulas.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Module.finrank_tensorProduct, Subspace.dual_finrank_eq]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 0 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], True),
    (['ℚ', 'ℚ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
    (['ℚ', '(Fin 2 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
    (['ℚ', '(Fin 3 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
