# name: la-card-fin-prod-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.TensorProducts.card_fin_prod file "la-card-fin-prod-test.sage"
# description: |Fin m × Fin n| = m n at seeded sizes (0 and 1 among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
cases = default_cases(universal, Sampler(SEED), count=6)
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
