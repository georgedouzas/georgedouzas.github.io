# name: la-finrank-functions-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaceResults.finrank_functions file "la-finrank-functions-test.sage"
# description: the functions on a finite set X have dimension |X|, at X = ∅ (Fin 0), Bool, Fin 5 and Fin 2 × Fin 3.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['(Fin 0)', 'inferInstance'], True),
    (['Bool', 'inferInstance'], False),
    (['(Fin 5)', 'inferInstance'], False),
    (['(Fin 2 × Fin 3)', 'inferInstance'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
