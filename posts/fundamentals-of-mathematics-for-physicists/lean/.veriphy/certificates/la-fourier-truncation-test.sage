# name: la-fourier-truncation-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Dimension.fourierModes_truncation_linearIndependent file "la-fourier-truncation-test.sage"
# description: the modes up to N are independent at N = 0, 1, 2, 3 (the restriction of the independent family of all modes).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'exact fourierModes_linearIndependent.comp _ (Sum.map_injective.mpr ⟨Fin.val_injective, Fin.val_injective⟩)'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['1'], False),
    (['2'], False),
    (['3'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
