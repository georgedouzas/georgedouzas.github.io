# name: la-card-fin-prod3-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.card_fin_prod3 file "la-card-fin-prod3-test.sage"
# description: |Fin a × Fin b × Fin c| = a b c at seeded sizes (0 and 1 among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
cases = default_cases(universal, Sampler(SEED), count=6)
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
