# name: set-finer-coarser-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.FiniteTopologies.finer_coarser
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'ℝ', '⊤', '⊥', 'inferInstance', '⊥', 'id', 'bot_le', 'bot_le'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) (t₁ t₂ : TopologicalSpace X) (s₁ s₂ : TopologicalSpace Y) (f : X → Y) (ht : t₂ ≤ t₁) (hs : s₂ ≤ s₁), (Continuous[t₁, s₁] f → Continuous[t₂, s₁] f) ∧ (Continuous[t₁, s₂] f → Continuous[t₁, s₁] f) := by\n  intro X Y t₁ t₂ s₁ s₂ f ht hs\n  exact (⟨fun h => continuous_le_dom ht h, fun h => continuous_le_rng hs h⟩)\nexact @key (ℝ) (ℝ) (⊤) (⊥) (inferInstance) (⊥) (id) (bot_le) (bot_le)'),
    (['Unit', 'Unit', '⊥', '⊥', '⊥', '⊥', 'id', 'le_rfl', 'le_rfl'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) (t₁ t₂ : TopologicalSpace X) (s₁ s₂ : TopologicalSpace Y) (f : X → Y) (ht : t₂ ≤ t₁) (hs : s₂ ≤ s₁), (Continuous[t₁, s₁] f → Continuous[t₂, s₁] f) ∧ (Continuous[t₁, s₂] f → Continuous[t₁, s₁] f) := by\n  intro X Y t₁ t₂ s₁ s₂ f ht hs\n  exact (⟨fun h => continuous_le_dom ht h, fun h => continuous_le_rng hs h⟩)\nexact @key (Unit) (Unit) (⊥) (⊥) (⊥) (⊥) (id) (le_rfl) (le_rfl)'),
    (['Bool', 'Bool', '⊤', '⊤', '⊤', '⊥', 'not', 'le_rfl', 'bot_le'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) (t₁ t₂ : TopologicalSpace X) (s₁ s₂ : TopologicalSpace Y) (f : X → Y) (ht : t₂ ≤ t₁) (hs : s₂ ≤ s₁), (Continuous[t₁, s₁] f → Continuous[t₂, s₁] f) ∧ (Continuous[t₁, s₂] f → Continuous[t₁, s₁] f) := by\n  intro X Y t₁ t₂ s₁ s₂ f ht hs\n  exact (⟨fun h => continuous_le_dom ht h, fun h => continuous_le_rng hs h⟩)\nexact @key (Bool) (Bool) (⊤) (⊤) (⊤) (⊥) (not) (le_rfl) (bot_le)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
