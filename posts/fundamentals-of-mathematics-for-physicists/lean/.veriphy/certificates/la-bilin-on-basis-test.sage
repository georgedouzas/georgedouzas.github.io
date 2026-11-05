# name: la-bilin-on-basis-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.BilinearForms.bilin_on_basis file "la-bilin-on-basis-test.sage"
# description: B(e_i, e_j) = B_{ij} for explicit matrices at diagonal and off-diagonal pairs, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = "norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]"  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['1', '!![5]', '0', '0'], True),
    (['2', '!![1, 2; -1, 0]', '1', '0'], False),
    (['3', '!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]', '1', '2'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
