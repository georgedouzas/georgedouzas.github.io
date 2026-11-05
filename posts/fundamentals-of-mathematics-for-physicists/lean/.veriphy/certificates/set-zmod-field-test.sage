# name: set-zmod-field-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.zmod_prime_field
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', '(by norm_num)', '2', '(by decide)'], False, 'exact ⟨2, by decide⟩'),
    (['2', '(by norm_num)', '1', '(by decide)'], True, 'exact ⟨1, by decide⟩'),
    (['5', '(by norm_num)', '3', '(by decide)'], False, 'exact ⟨2, by decide⟩'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
