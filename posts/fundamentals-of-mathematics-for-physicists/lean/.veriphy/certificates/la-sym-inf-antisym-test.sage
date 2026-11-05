# name: la-sym-inf-antisym-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Dimension.sym_inf_antisym file "la-sym-inf-antisym-test.sage"
# description: a matrix both symmetric and antisymmetric vanishes, at n = 0, 1, 2, 3.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rw [eq_bot_iff]\nrintro A ⟨hs, ha⟩\nchange Aᵀ = A at hs\nchange Aᵀ = -A at ha\nrw [Submodule.mem_bot]\next i j\nhave h1 := congrFun (congrFun hs i) j\nhave h2 := congrFun (congrFun ha i) j\nsimp at h1 h2 ⊢\nlinarith'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['1'], False),
    (['2'], False),
    (['3'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
