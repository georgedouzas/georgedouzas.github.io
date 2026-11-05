# name: set-quotient-compact-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Compact.quotient_compact
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'inferInstance', 'inferInstance', '⊤'], False, 'exact Quotient.compactSpace'),
    (['Unit', 'inferInstance', 'inferInstance', '⊥'], True, 'exact Quotient.compactSpace'),
    (['Set.Icc (0 : ℝ) 1', 'inferInstance', '(isCompact_iff_compactSpace.1 isCompact_Icc)', '⊤'], False, 'exact Quotient.compactSpace'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
