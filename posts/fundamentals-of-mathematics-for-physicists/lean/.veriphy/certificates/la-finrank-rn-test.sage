# name: la-finrank-rn-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Dimension.finrank_rn file "la-finrank-rn-test.sage"
# description: dim ℝⁿ = n at seeded n (0 and 1 among them), by the product formula.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
