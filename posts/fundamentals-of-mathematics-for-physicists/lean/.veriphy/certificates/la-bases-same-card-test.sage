# name: la-bases-same-card-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Dimension.bases_same_card file "la-bases-same-card-test.sage"
# description: two bases of the same space have equally many elements, at ℚ⁰, ℚ² (indexed by Fin 2 and by Bool) and ℚ³.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 0 → ℚ)', '(Fin 0)', '(Fin 0)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 0)', 'Pi.basisFun ℚ (Fin 0)'], True),
    (['ℚ', '(Fin 2 → ℚ)', '(Fin 2)', 'Bool', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 2)', '(Pi.basisFun ℚ (Fin 2)).reindex finTwoEquiv'], False),
    (['ℚ', '(Fin 3 → ℚ)', '(Fin 3)', '(Fin 3)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 3)', '(Pi.basisFun ℚ (Fin 3)).reindex (Equiv.swap 0 2)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
