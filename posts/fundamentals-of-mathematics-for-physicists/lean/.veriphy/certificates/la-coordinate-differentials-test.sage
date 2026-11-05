# name: la-coordinate-differentials-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualBasis.coordinate_differentials file "la-coordinate-differentials-test.sage"
# description: dx^i(∂_j) = δ^i_j in ℝⁿ at n = 1, 2, 3 for diagonal and off-diagonal pairs, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Pi.single_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['1', '0', '0'], True),
    (['2', '0', '1'], False),
    (['3', '2', '2'], False),
    (['3', '1', '0'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
