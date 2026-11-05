# name: set-subspace-transitive-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Subspace.subspace_transitive
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'inferInstance', 'Set.univ', 'Set.Icc 0 1', '(Set.subset_univ _)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (Y Z : Set X) (h : Z ⊆ Y), TopologicalSpace.induced (Set.inclusion h) (inferInstance : TopologicalSpace Y) = (inferInstance : TopologicalSpace Z) := by\n  intro X inst1 Y Z h\n  change TopologicalSpace.induced (Set.inclusion h) (TopologicalSpace.induced Subtype.val _) =\n    TopologicalSpace.induced Subtype.val _\n  rw [induced_compose]\n  rfl\nexact @key (ℝ) (inferInstance) (Set.univ) (Set.Icc 0 1) ((Set.subset_univ _))'),
    (['ℝ', 'inferInstance', '∅', '∅', 'le_rfl'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (Y Z : Set X) (h : Z ⊆ Y), TopologicalSpace.induced (Set.inclusion h) (inferInstance : TopologicalSpace Y) = (inferInstance : TopologicalSpace Z) := by\n  intro X inst1 Y Z h\n  change TopologicalSpace.induced (Set.inclusion h) (TopologicalSpace.induced Subtype.val _) =\n    TopologicalSpace.induced Subtype.val _\n  rw [induced_compose]\n  rfl\nexact @key (ℝ) (inferInstance) (∅) (∅) (le_rfl)'),
    (['ℝ × ℝ', 'inferInstance', 'Set.univ', 'KaluzaKlein.Sets.Sets.circle', '(Set.subset_univ _)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (Y Z : Set X) (h : Z ⊆ Y), TopologicalSpace.induced (Set.inclusion h) (inferInstance : TopologicalSpace Y) = (inferInstance : TopologicalSpace Z) := by\n  intro X inst1 Y Z h\n  change TopologicalSpace.induced (Set.inclusion h) (TopologicalSpace.induced Subtype.val _) =\n    TopologicalSpace.induced Subtype.val _\n  rw [induced_compose]\n  rfl\nexact @key (ℝ × ℝ) (inferInstance) (Set.univ) (KaluzaKlein.Sets.Sets.circle) ((Set.subset_univ _))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
