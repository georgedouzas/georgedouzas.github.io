# name: la-finrank-antisym-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym file "la-finrank-antisym-test.sage"
# description: dim so(n) = n(n-1)/2 at n = 0, 1, 2, 3, 4: the strictly upper-triangular coordinates, counted by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rw [(antisymEquiv _).finrank_eq, Module.finrank_fintype_fun_eq_card]\ndecide'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['1'], False),
    (['2'], False),
    (['3'], False),
    (['4'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
