# name: la-dualbasis-delta-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualSpace.dualBasis_delta file "la-dualbasis-delta-test.sage"
# description: e^i(e_j) = δ^i_j for the standard bases of ℚ¹, ℚ², ℚ³ at diagonal and off-diagonal pairs, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Basis.dualBasis_apply, Pi.basisFun_repr, Pi.single_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 1 → ℚ)', '(Fin 1)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 1)', '0', '0'], True),
    (['ℚ', '(Fin 2 → ℚ)', '(Fin 2)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 2)', '0', '1'], False),
    (['ℚ', '(Fin 3 → ℚ)', '(Fin 3)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 3)', '2', '2'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
