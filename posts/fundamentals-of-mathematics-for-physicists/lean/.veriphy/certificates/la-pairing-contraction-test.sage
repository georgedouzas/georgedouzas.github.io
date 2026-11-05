# name: la-pairing-contraction-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Contraction.pairing_is_contraction file "la-pairing-contraction-test.sage"
# description: tr (v αᵀ) = α_i v^i at n = 0 and explicit vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'norm_num [Matrix.trace, Matrix.vecMulVec_apply, dotProduct, Fin.sum_univ_succ]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]', '![]'], True, 'simp [Matrix.trace]'),
    (['3', '![1, 2, 3]', '![1, 0, -1]'], False),
    (['2', '![1 / 2, 4]', '![2, -1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
