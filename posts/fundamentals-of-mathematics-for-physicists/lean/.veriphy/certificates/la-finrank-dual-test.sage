# name: la-finrank-dual-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual file "la-finrank-dual-test.sage"
# description: dim V* = dim V at V = ℚ⁰, ℚ², ℚ³ and ℚ, by the finrank formulas.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Subspace.dual_finrank_eq]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 0 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], True),
    (['ℚ', '(Fin 2 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
    (['ℚ', '(Fin 3 → ℚ)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
    (['ℚ', 'ℚ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
