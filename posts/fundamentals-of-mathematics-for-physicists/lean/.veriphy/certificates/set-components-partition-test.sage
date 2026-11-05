# name: set-components-partition-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.components_partition
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'inferInstance', '0', '1'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (x y : X), x ∈ connectedComponent x ∧ (connectedComponent x = connectedComponent y ∨ Disjoint (connectedComponent x) (connectedComponent y)) := by\n  intro X inst1 x y\n  refine ⟨mem_connectedComponent, ?_⟩\n  by_cases h : y ∈ connectedComponent x\n  · left; exact (connectedComponent_eq h)\n  · right; exact connectedComponent_disjoint (fun h' => h (h' ▸ mem_connectedComponent))\nexact @key (ℝ) (inferInstance) (0) (1)"),
    (['Unit', 'inferInstance', '()', '()'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (x y : X), x ∈ connectedComponent x ∧ (connectedComponent x = connectedComponent y ∨ Disjoint (connectedComponent x) (connectedComponent y)) := by\n  intro X inst1 x y\n  refine ⟨mem_connectedComponent, ?_⟩\n  by_cases h : y ∈ connectedComponent x\n  · left; exact (connectedComponent_eq h)\n  · right; exact connectedComponent_disjoint (fun h' => h (h' ▸ mem_connectedComponent))\nexact @key (Unit) (inferInstance) (()) (())"),
    (['ℤ', 'inferInstance', '0', '1'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (x y : X), x ∈ connectedComponent x ∧ (connectedComponent x = connectedComponent y ∨ Disjoint (connectedComponent x) (connectedComponent y)) := by\n  intro X inst1 x y\n  refine ⟨mem_connectedComponent, ?_⟩\n  by_cases h : y ∈ connectedComponent x\n  · left; exact (connectedComponent_eq h)\n  · right; exact connectedComponent_disjoint (fun h' => h (h' ▸ mem_connectedComponent))\nexact @key (ℤ) (inferInstance) (0) (1)"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
