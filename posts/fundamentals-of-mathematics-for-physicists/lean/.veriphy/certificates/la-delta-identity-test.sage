# name: la-delta-identity-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MixedTensors.delta_identity file "la-delta-identity-test.sage"
# description: δ^i_j v^j = v^i at n = 0 and explicit vectors, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext i\nfin_cases i <;> simp [Matrix.mulVec, dotProduct, Fin.sum_univ_succ, Matrix.one_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]'], True, 'ext i\nexact i.elim0'),
    (['2', '![1, 2]'], False),
    (['3', '![1 / 2, 0, -3]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
