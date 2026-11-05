# name: set-card-topologies-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.FiniteTopologies.card_topologies
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 0', 'inferInstance', 'inferInstance'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α : Type) [inst1 : Fintype α] [inst2 : DecidableEq α], Nat.card (TopologicalSpace α) = (Finset.univ.filter fun F : Finset (Finset α) => IsFinTop F).card := by\n  intro α inst1 inst2\n  rw [Nat.card_congr (topEquiv α), Nat.card_eq_fintype_card, Fintype.card_subtype]\nexact @key (Fin 0) (inferInstance) (inferInstance)'),
    (['Fin 1', 'inferInstance', 'inferInstance'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α : Type) [inst1 : Fintype α] [inst2 : DecidableEq α], Nat.card (TopologicalSpace α) = (Finset.univ.filter fun F : Finset (Finset α) => IsFinTop F).card := by\n  intro α inst1 inst2\n  rw [Nat.card_congr (topEquiv α), Nat.card_eq_fintype_card, Fintype.card_subtype]\nexact @key (Fin 1) (inferInstance) (inferInstance)'),
    (['Fin 2', 'inferInstance', 'inferInstance'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (α : Type) [inst1 : Fintype α] [inst2 : DecidableEq α], Nat.card (TopologicalSpace α) = (Finset.univ.filter fun F : Finset (Finset α) => IsFinTop F).card := by\n  intro α inst1 inst2\n  rw [Nat.card_congr (topEquiv α), Nat.card_eq_fintype_card, Fintype.card_subtype]\nexact @key (Fin 2) (inferInstance) (inferInstance)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
