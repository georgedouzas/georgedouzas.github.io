# name: la-finrank-dual-rn-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualSpace.finrank_dual_rn file "la-finrank-dual-rn-test.sage"
# description: dim (ℝⁿ)* = n at seeded n (0 and 1 among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Subspace.dual_finrank_eq]'  # EDIT: the tactic block proving the claim at one instance
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
