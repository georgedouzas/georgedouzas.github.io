# name: set-zmod-cyclic-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.zmod_cyclic
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['6', '4'], False, 'exact ⟨4, by decide⟩'),
    (['1', '0'], True, 'exact ⟨0, by decide⟩'),
    (['3', '2'], False, 'exact ⟨2, by decide⟩'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
