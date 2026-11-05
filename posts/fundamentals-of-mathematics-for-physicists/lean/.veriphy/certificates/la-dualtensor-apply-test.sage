# name: la-dualtensor-apply-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.TensorProducts.dualTensor_apply file "la-dualtensor-apply-test.sage"
# description: α ⊗ v acts as w ↦ α(w) v, at the zero covector and coordinate covectors of ℝ² and ℝ³, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [dualTensorHom_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', '0', '![1, 2]', '![3, -1]'], True),
    (['(Fin 2 → ℝ)', 'inferInstance', 'inferInstance', 'LinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 0', '![1, 2]', '![3, -1]'], False),
    (['(Fin 3 → ℝ)', 'inferInstance', 'inferInstance', 'LinearMap.proj (R := ℝ) (φ := fun _ : Fin 3 => ℝ) 2', '![1, 0, -1]', '![1, 2, 3]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
