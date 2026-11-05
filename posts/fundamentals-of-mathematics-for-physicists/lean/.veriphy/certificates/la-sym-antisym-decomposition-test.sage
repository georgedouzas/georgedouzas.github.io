# name: la-sym-antisym-decomposition-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Dimension.sym_antisym_decomposition file "la-sym-antisym-decomposition-test.sage"
# description: A = ½(A + Aᵀ) + ½(A − Aᵀ) with symmetric and antisymmetric parts, at n = 0, 2, 3, entry by entry.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'refine ⟨?_, ?_, ?_⟩\n· ext i j; fin_cases i <;> fin_cases j <;> norm_num\n· show Matrix.transpose _ = _; ext i j; fin_cases i <;> fin_cases j <;> norm_num\n· show Matrix.transpose _ = -_; ext i j; fin_cases i <;> fin_cases j <;> norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0'], True),
    (['2', '!![1, 2; 3, 4]'], False),
    (['3', '!![1, 0, 2; -1, 3, 5; 4, 4, 0]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
