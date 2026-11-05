# name: set-inverse-comp-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Composition.inverse_comp
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', 'Fin 3', 'Equiv.refl _'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (e : X ≃ Y), e.symm ∘ e = id ∧ e ∘ e.symm = id := by\n  intro X Y e\n  exact (⟨e.symm_comp_self, e.self_comp_symm⟩)\nexact @key (Fin 3) (Fin 3) (Equiv.refl _)'),
    (['Fin 3', 'Fin 3', 'finRotate 3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (e : X ≃ Y), e.symm ∘ e = id ∧ e ∘ e.symm = id := by\n  intro X Y e\n  exact (⟨e.symm_comp_self, e.self_comp_symm⟩)\nexact @key (Fin 3) (Fin 3) (finRotate 3)'),
    (['ℤ', 'ℤ', 'Equiv.addRight 1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (e : X ≃ Y), e.symm ∘ e = id ∧ e ∘ e.symm = id := by\n  intro X Y e\n  exact (⟨e.symm_comp_self, e.self_comp_symm⟩)\nexact @key (ℤ) (ℤ) (Equiv.addRight 1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
