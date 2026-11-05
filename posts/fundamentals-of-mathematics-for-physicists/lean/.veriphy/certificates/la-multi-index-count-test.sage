# name: la-multi-index-count-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.TensorProducts.multi_index_count file "la-multi-index-count-test.sage"
# description: a tensor with k indices in dimension n has nᵏ components, at seeded n, k (0 and 1 among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
cases = default_cases(universal, Sampler(SEED), count=6)
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
