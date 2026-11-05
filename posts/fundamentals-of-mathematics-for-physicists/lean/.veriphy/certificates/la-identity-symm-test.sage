# name: la-identity-symm-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.SymmetricForms.identity_symm file "la-identity-symm-test.sage"
# description: the identity matrix is symmetric, at seeded n (0 and 1 among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext i j\nsimp [Matrix.one_apply, eq_comm]'  # EDIT: entry by entry
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
