# name: la-symmetric-iff-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.SymmetricForms.symmetric_iff file "la-symmetric-iff-test.sage"
# description: B(v, w) = B(w, v) for all v, w exactly when Bᵀ = B, at symmetric 2 × 2 and 3 × 3 matrices and the zero matrix (both sides true), the identity checked entry by entry and as a polynomial identity in v, w.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = "constructor\n· intro _\n  ext i j\n  fin_cases i <;> fin_cases j <;> norm_num\n· intro _ v w\n  simp [Matrix.toBilin'_apply', Matrix.mulVec, dotProduct, Fin.sum_univ_succ] <;> ring"  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['2', '0'], True),
    (['2', '!![1, 2; 2, 4]'], False),
    (['3', '!![2, -3 / 2, 0; -3 / 2, 1, 2; 0, 2, 0]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
