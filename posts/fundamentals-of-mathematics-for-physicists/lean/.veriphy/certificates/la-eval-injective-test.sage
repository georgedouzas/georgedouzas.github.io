# name: la-eval-injective-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualSpace.eval_injective file "la-eval-injective-test.sage"
# description: v ↦ ev_v is injective at V = ℚ⁰, ℚ², ℚ³ (the coordinate functionals separate vectors).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'intro v w h\nfunext i\nsimpa using LinearMap.congr_fun h (LinearMap.proj (R := ℚ) (φ := fun _ => ℚ) i)'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 0 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance'], True),
    (['ℚ', '(Fin 2 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance'], False),
    (['ℚ', '(Fin 3 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
