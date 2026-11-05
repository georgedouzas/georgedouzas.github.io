# name: la-bilin-components-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.BilinearForms.bilin_components file "la-bilin-components-test.sage"
# description: B(v, w) = B_{ij} v^i w^j for explicit matrices and vectors (n = 0, 2, 3), by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = "norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]"  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '![]', '![]'], True),
    (['2', '!![1, 2; -1, 0]', '![1, 3]', '![2, -1]'], False),
    (['3', '!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]', '![1, 2, -1]', '![0, 1, 1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
