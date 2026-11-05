# name: set-finite-hausdorff-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Hausdorff.finite_hausdorff_discrete
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'inferInstance', '⊥', '(@DiscreteTopology.toT2Space _ ⊥ ⟨rfl⟩)'], False, 'exact ⟨rfl⟩'),
    (['Fin 1', 'inferInstance', '⊥', '(@DiscreteTopology.toT2Space _ ⊥ ⟨rfl⟩)'], True, 'exact ⟨rfl⟩'),
    (['Bool', 'inferInstance', '⊥', '(@DiscreteTopology.toT2Space _ ⊥ ⟨rfl⟩)'], False, 'exact ⟨rfl⟩'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
