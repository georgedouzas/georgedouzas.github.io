# name: set-graph-connected-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.graph_connected_iff
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '⊥'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (G : SimpleGraph (Fin n)), G.Connected ↔ Nonempty (Fin n) ∧ ∀ u v, G.Reachable u v := by\n  intro n G\n  exact (⟨fun h => ⟨h.nonempty, h.preconnected⟩, fun ⟨h1, h2⟩ => ⟨h2⟩⟩)\nexact @key (0) (⊥)'),
    (['3', '⊤'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (G : SimpleGraph (Fin n)), G.Connected ↔ Nonempty (Fin n) ∧ ∀ u v, G.Reachable u v := by\n  intro n G\n  exact (⟨fun h => ⟨h.nonempty, h.preconnected⟩, fun ⟨h1, h2⟩ => ⟨h2⟩⟩)\nexact @key (3) (⊤)'),
    (['5', 'KaluzaKlein.Sets.Connected.graphABCDE'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (G : SimpleGraph (Fin n)), G.Connected ↔ Nonempty (Fin n) ∧ ∀ u v, G.Reachable u v := by\n  intro n G\n  exact (⟨fun h => ⟨h.nonempty, h.preconnected⟩, fun ⟨h1, h2⟩ => ⟨h2⟩⟩)\nexact @key (5) (KaluzaKlein.Sets.Connected.graphABCDE)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
