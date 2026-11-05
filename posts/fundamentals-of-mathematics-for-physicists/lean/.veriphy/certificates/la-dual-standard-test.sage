# name: la-dual-standard-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualResults.dual_standard file "la-dual-standard-test.sage"
# description: the dual basis of the standard basis is the standard basis (1⁻¹ = 1) at seeded n (0 and 1 among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'exact Matrix.inv_eq_left_inv (by simp)'  # EDIT: 1 * 1 = 1 checks the inverse
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
