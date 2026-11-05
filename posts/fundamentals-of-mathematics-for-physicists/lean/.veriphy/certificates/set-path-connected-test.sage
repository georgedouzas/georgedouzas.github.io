# name: set-path-connected-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.pathConnected_connected
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'inferInstance', 'Set.univ', 'isPathConnected_univ'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (S : Set X) (h : IsPathConnected S), IsConnected S := by\n  intro X inst1 S h\n  exact (h.isConnected)\nexact @key (ℝ) (inferInstance) (Set.univ) (isPathConnected_univ)'),
    (['ℝ', 'inferInstance', 'Set.Icc 0 1', '((convex_Icc 0 1).isPathConnected (Set.nonempty_Icc.2 zero_le_one))'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (S : Set X) (h : IsPathConnected S), IsConnected S := by\n  intro X inst1 S h\n  exact (h.isConnected)\nexact @key (ℝ) (inferInstance) (Set.Icc 0 1) (((convex_Icc 0 1).isPathConnected (Set.nonempty_Icc.2 zero_le_one)))'),
    (['Fin 3 → ℝ', 'inferInstance', 'Set.univ', 'isPathConnected_univ'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type) [inst1 : TopologicalSpace X] (S : Set X) (h : IsPathConnected S), IsConnected S := by\n  intro X inst1 S h\n  exact (h.isConnected)\nexact @key (Fin 3 → ℝ) (inferInstance) (Set.univ) (isPathConnected_univ)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
