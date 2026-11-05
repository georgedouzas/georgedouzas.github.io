# name: la-mem-ker-a-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearMaps.mem_ker_A file "la-mem-ker-a-test.sage"
# description: x ∈ ker A ↔ x + 2y + 3z = 0 at the origin, at two kernel vectors and at a vector outside, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [KaluzaKlein.Linear.A, LinearMap.mem_ker, funext_iff, Fin.forall_fin_succ, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]\n<;> norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['![1, 1, -1]'], False),
    (['![-3, 0, 1]'], False),
    (['![1, 0, 0]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
