# name: set-finite-map-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Maps.finite_map_image_preimage
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', '0', '1', '(by decide)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Set.range ![x, x, y] = {x, y} ∧ ![x, x, y] ⁻¹' {x} = {0, 1} := by\n  intro β x y hxy\n  constructor\n  · ext z\n    simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]\n    constructor\n    · rintro ⟨i, rfl⟩; fin_cases i <;> simp\n    · rintro (rfl | rfl)\n      · exact ⟨0, rfl⟩\n      · exact ⟨2, rfl⟩\n  · ext i\n    fin_cases i <;> simp [hxy.symm]\nexact @key (Fin 2) (0) (1) ((by decide))"),
    (['ℕ', '5', '7', '(by decide)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Set.range ![x, x, y] = {x, y} ∧ ![x, x, y] ⁻¹' {x} = {0, 1} := by\n  intro β x y hxy\n  constructor\n  · ext z\n    simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]\n    constructor\n    · rintro ⟨i, rfl⟩; fin_cases i <;> simp\n    · rintro (rfl | rfl)\n      · exact ⟨0, rfl⟩\n      · exact ⟨2, rfl⟩\n  · ext i\n    fin_cases i <;> simp [hxy.symm]\nexact @key (ℕ) (5) (7) ((by decide))"),
    (['Bool', 'true', 'false', '(by decide)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Set.range ![x, x, y] = {x, y} ∧ ![x, x, y] ⁻¹' {x} = {0, 1} := by\n  intro β x y hxy\n  constructor\n  · ext z\n    simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]\n    constructor\n    · rintro ⟨i, rfl⟩; fin_cases i <;> simp\n    · rintro (rfl | rfl)\n      · exact ⟨0, rfl⟩\n      · exact ⟨2, rfl⟩\n  · ext i\n    fin_cases i <;> simp [hxy.symm]\nexact @key (Bool) (true) (false) ((by decide))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
