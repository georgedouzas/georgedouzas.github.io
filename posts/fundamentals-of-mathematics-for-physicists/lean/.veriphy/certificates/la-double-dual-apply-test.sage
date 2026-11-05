# name: la-double-dual-apply-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualSpace.doubleDual_apply file "la-double-dual-apply-test.sage"
# description: ev_v(α) = α(v) at the zero vector and explicit vectors and functionals of ℚ², by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rfl'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 2 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', '0', '0'], True),
    (['ℚ', '(Fin 2 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', '![1, 2]', 'LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1'], False),
    (['ℚ', '(Fin 3 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', '![1, 0, -1]', 'LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
