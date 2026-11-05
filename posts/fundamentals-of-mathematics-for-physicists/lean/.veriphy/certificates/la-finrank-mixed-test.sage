# name: la-finrank-mixed-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MixedTensors.finrank_mixed file "la-finrank-mixed-test.sage"
# description: a type (r, s) tensor in dimension n has n^(r+s) components, at seeded n, r, s (0 and 1 among them).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = ('simp [MixedTensor, Module.finrank_pi_fintype, Module.finrank_fintype_fun_eq_card, Fintype.card_fun, '
         'Module.finrank_self]')  # EDIT: the product formula for the components
cases = default_cases(universal, Sampler(SEED), count=6)
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in cases]
