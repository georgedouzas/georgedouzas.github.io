# name: set-maps-triples-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Counting.maps_three_as_triples
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (β : Type), Function.Bijective (fun f : Fin 3 → β => (f 0, f 1, f 2)) := by\n  intro β\n  refine ⟨fun f g h => ?_, fun ⟨a, b, c⟩ => ⟨![a, b, c], rfl⟩⟩\n  simp only [Prod.mk.injEq] at h\n  funext i; fin_cases i\n  · exact h.1\n  · exact h.2.1\n  · exact h.2.2\nexact @key (Fin 2)'),
    (['Unit'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (β : Type), Function.Bijective (fun f : Fin 3 → β => (f 0, f 1, f 2)) := by\n  intro β\n  refine ⟨fun f g h => ?_, fun ⟨a, b, c⟩ => ⟨![a, b, c], rfl⟩⟩\n  simp only [Prod.mk.injEq] at h\n  funext i; fin_cases i\n  · exact h.1\n  · exact h.2.1\n  · exact h.2.2\nexact @key (Unit)'),
    (['ℕ'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (β : Type), Function.Bijective (fun f : Fin 3 → β => (f 0, f 1, f 2)) := by\n  intro β\n  refine ⟨fun f g h => ?_, fun ⟨a, b, c⟩ => ⟨![a, b, c], rfl⟩⟩\n  simp only [Prod.mk.injEq] at h\n  funext i; fin_cases i\n  · exact h.1\n  · exact h.2.1\n  · exact h.2.2\nexact @key (ℕ)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
