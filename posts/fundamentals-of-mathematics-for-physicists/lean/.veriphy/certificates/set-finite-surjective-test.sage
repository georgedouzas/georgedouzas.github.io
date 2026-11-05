# name: set-finite-surjective-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Surjective.finite_map_surjective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', '0', '1', '(by decide)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (h : ∀ b, b = x ∨ b = y), Function.Surjective ![x, x, y] := by\n  intro β x y h\n  intro b\n  rcases h b with rfl | rfl\n  · exact ⟨0, rfl⟩\n  · exact ⟨2, rfl⟩\nexact @key (Fin 2) (0) (1) ((by decide))'),
    (['Bool', 'true', 'false', '(by decide)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (h : ∀ b, b = x ∨ b = y), Function.Surjective ![x, x, y] := by\n  intro β x y h\n  intro b\n  rcases h b with rfl | rfl\n  · exact ⟨0, rfl⟩\n  · exact ⟨2, rfl⟩\nexact @key (Bool) (true) (false) ((by decide))'),
    (['Fin 2', '1', '0', '(by decide)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (h : ∀ b, b = x ∨ b = y), Function.Surjective ![x, x, y] := by\n  intro β x y h\n  intro b\n  rcases h b with rfl | rfl\n  · exact ⟨0, rfl⟩\n  · exact ⟨2, rfl⟩\nexact @key (Fin 2) (1) (0) ((by decide))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
