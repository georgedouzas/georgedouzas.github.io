# name: la-dot-product-form-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.BilinearForms.dot_product_form file "la-dot-product-form-test.sage"
# description: the form of the identity matrix is the dot product, at n = 0, 2, 3 with explicit vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = "norm_num [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Pi.single_apply]"  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]', '![]'], True),
    (['2', '![1, 3]', '![2, -1]'], False),
    (['3', '![1, 2, -1]', '![1 / 2, 0, 4]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
