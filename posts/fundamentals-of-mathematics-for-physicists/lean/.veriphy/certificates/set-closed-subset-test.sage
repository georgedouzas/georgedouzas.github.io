# name: set-closed-subset-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Compact.closed_subset_compact
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ × ℝ', 'inferInstance', 'KaluzaKlein.Sets.Sets.closedDisc', 'KaluzaKlein.Sets.Sets.circle', 'KaluzaKlein.Sets.Compact.closedDisc_compact', 'KaluzaKlein.Sets.Compact.circle_closed_bounded.1', '(fun p hp => le_of_eq hp)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K C : Set E) (hK : IsCompact K) (hC : IsClosed C) (hCK : C ⊆ K), IsCompact C := by\n  intro E inst1 K C hK hC hCK\n  exact (hK.of_isClosed_subset hC hCK)\nexact @key (ℝ × ℝ) (inferInstance) (KaluzaKlein.Sets.Sets.closedDisc) (KaluzaKlein.Sets.Sets.circle) (KaluzaKlein.Sets.Compact.closedDisc_compact) (KaluzaKlein.Sets.Compact.circle_closed_bounded.1) ((fun p hp => le_of_eq hp))'),
    (['ℝ', 'inferInstance', 'Set.Icc 0 1', '∅', 'isCompact_Icc', 'isClosed_empty', '(Set.empty_subset _)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K C : Set E) (hK : IsCompact K) (hC : IsClosed C) (hCK : C ⊆ K), IsCompact C := by\n  intro E inst1 K C hK hC hCK\n  exact (hK.of_isClosed_subset hC hCK)\nexact @key (ℝ) (inferInstance) (Set.Icc 0 1) (∅) (isCompact_Icc) (isClosed_empty) ((Set.empty_subset _))'),
    (['ℝ', 'inferInstance', 'Set.Icc 0 2', 'Set.Icc 0 1', 'isCompact_Icc', 'isClosed_Icc', '(Set.Icc_subset_Icc le_rfl (by norm_num))'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (E : Type) [inst1 : TopologicalSpace E] (K C : Set E) (hK : IsCompact K) (hC : IsClosed C) (hCK : C ⊆ K), IsCompact C := by\n  intro E inst1 K C hK hC hCK\n  exact (hK.of_isClosed_subset hC hCK)\nexact @key (ℝ) (inferInstance) (Set.Icc 0 2) (Set.Icc 0 1) (isCompact_Icc) (isClosed_Icc) ((Set.Icc_subset_Icc le_rfl (by norm_num)))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
