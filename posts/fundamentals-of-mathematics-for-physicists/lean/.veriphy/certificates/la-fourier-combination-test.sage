# name: la-fourier-combination-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearCombinations.fourier_combination file "la-fourier-combination-test.sage"
# description: a cos φ + b sin φ is the combination a • cos + b • sin, at seeded scalars (zero and one among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'funext φ\nsimp'  # EDIT: the tactic block proving the claim at one instance
cases = default_cases(universal, Sampler(SEED))
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
