# name: la-tensorbasis-apply-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.TensorProducts.tensorBasis_apply file "la-tensorbasis-apply-test.sage"
# description: the basis vector (i, a) of V ⊗ W is e_i ⊗ f_a, for the standard bases of ℚ¹ ⊗ ℚ¹, ℚ² ⊗ ℚ³ and ℚ³ ⊗ ℚ².
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [tensorBasis]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 1 → ℚ)', '(Fin 1 → ℚ)', '(Fin 1)', '(Fin 1)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 1)', 'Pi.basisFun ℚ (Fin 1)', '0', '0'], True),
    (['ℚ', '(Fin 2 → ℚ)', '(Fin 3 → ℚ)', '(Fin 2)', '(Fin 3)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 2)', 'Pi.basisFun ℚ (Fin 3)', '1', '2'], False),
    (['ℚ', '(Fin 3 → ℚ)', '(Fin 2 → ℚ)', '(Fin 3)', '(Fin 2)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 3)', 'Pi.basisFun ℚ (Fin 2)', '0', '1'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
