# name: la-finrank-matrix-sq-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_matrix_sq file "la-finrank-matrix-sq-test.sage"
# description: dim of the n × n matrices is n² at n = 0, 1, 2, 3 (finrank of matrices by the product formula).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [Module.finrank_matrix]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['1'], True),
    (['2'], False),
    (['3'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
