# name: set-homeomorph-open-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Homeomorphism.homeomorph_open_closed
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', 'Homeomorph.refl ℝ', 'Set.Ioo 0 1'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (f : X ≃ₜ Y) (U : Set X), (IsOpen U → IsOpen (f '' U)) ∧ (IsClosed U → IsClosed (f '' U)) := by\n  intro X Y inst1 inst2 f U\n  exact (⟨fun h => f.isOpenMap U h, fun h => f.isClosedMap U h⟩)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.refl ℝ) (Set.Ioo 0 1)"),
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', 'Homeomorph.neg ℝ', 'Set.Icc 0 1'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (f : X ≃ₜ Y) (U : Set X), (IsOpen U → IsOpen (f '' U)) ∧ (IsClosed U → IsClosed (f '' U)) := by\n  intro X Y inst1 inst2 f U\n  exact (⟨fun h => f.isOpenMap U h, fun h => f.isClosedMap U h⟩)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.neg ℝ) (Set.Icc 0 1)"),
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', 'Homeomorph.addRight 1', 'Set.univ'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (f : X ≃ₜ Y) (U : Set X), (IsOpen U → IsOpen (f '' U)) ∧ (IsClosed U → IsClosed (f '' U)) := by\n  intro X Y inst1 inst2 f U\n  exact (⟨fun h => f.isOpenMap U h, fun h => f.isClosedMap U h⟩)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Homeomorph.addRight 1) (Set.univ)"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
