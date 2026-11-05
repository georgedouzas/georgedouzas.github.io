# name: la-finrank-cotangent-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualSpace.finrank_cotangent file "la-finrank-cotangent-test.sage"
# description: the cotangent space of the model ℝⁿ at a point has dimension n, at n = 0, 2, 3.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rw [Subspace.dual_finrank_eq]\nshow Module.finrank ℝ (EuclideanSpace ℝ (Fin _)) = _\nsimp'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0'], True),
    (['2', '0'], False),
    (['3', 'EuclideanSpace.single 0 1'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
