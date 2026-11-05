# name: la-zero-degree-ne-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.zero_degree_ne file "la-zero-degree-ne-test.sage"
# description: the zero polynomial has degree ⊥, not n, at n = 0, 1, 2, 3.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['1'], False),
    (['2'], False),
    (['3'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
