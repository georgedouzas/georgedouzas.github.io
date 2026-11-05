# name: set-injective-iff-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.injective_iff_preimage_subsingleton
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', 'Fin 2', 'id'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Injective f ↔ ∀ y, (f ⁻¹' {y}).Subsingleton := by\n  intro X Y f\n  exact (⟨fun h _ _ ha _ hb => h (ha.trans hb.symm), fun h _ b hab => h (f b) hab rfl⟩)\nexact @key (Fin 2) (Fin 2) (id)"),
    (['Fin 3', 'ℕ', '![1, 2, 3]'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Injective f ↔ ∀ y, (f ⁻¹' {y}).Subsingleton := by\n  intro X Y f\n  exact (⟨fun h _ _ ha _ hb => h (ha.trans hb.symm), fun h _ b hab => h (f b) hab rfl⟩)\nexact @key (Fin 3) (ℕ) (![1, 2, 3])"),
    (['Fin 2', 'ℕ', '![0, 0]'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Injective f ↔ ∀ y, (f ⁻¹' {y}).Subsingleton := by\n  intro X Y f\n  exact (⟨fun h _ _ ha _ hb => h (ha.trans hb.symm), fun h _ b hab => h (f b) hab rfl⟩)\nexact @key (Fin 2) (ℕ) (![0, 0])"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
