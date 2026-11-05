# name: set-class-angle-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.class_angle_unique
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2 * Real.pi', '0', '7', '(by positivity)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T a φ : ℝ) (hT : 0 < T), ∃! ψ, ψ ∈ Set.Ico a (a + T) ∧ ∃ k : ℤ, ψ = φ + k * T := by\n  intro T a φ hT\n  refine ⟨toIcoMod hT a φ, ⟨toIcoMod_mem_Ico hT a φ, ⟨-toIcoDiv hT a φ, ?_⟩⟩, ?_⟩\n  · rw [toIcoMod]; push_cast; rw [zsmul_eq_mul]; ring\n  · rintro ψ ⟨hψ, k, rfl⟩\n    exact ((toIcoMod_eq_iff hT).2 ⟨hψ, -k, by rw [zsmul_eq_mul]; push_cast; ring⟩).symm\nexact @key (2 * Real.pi) (0) (7) ((by positivity))'),
    (['1', '0', '0', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T a φ : ℝ) (hT : 0 < T), ∃! ψ, ψ ∈ Set.Ico a (a + T) ∧ ∃ k : ℤ, ψ = φ + k * T := by\n  intro T a φ hT\n  refine ⟨toIcoMod hT a φ, ⟨toIcoMod_mem_Ico hT a φ, ⟨-toIcoDiv hT a φ, ?_⟩⟩, ?_⟩\n  · rw [toIcoMod]; push_cast; rw [zsmul_eq_mul]; ring\n  · rintro ψ ⟨hψ, k, rfl⟩\n    exact ((toIcoMod_eq_iff hT).2 ⟨hψ, -k, by rw [zsmul_eq_mul]; push_cast; ring⟩).symm\nexact @key (1) (0) (0) ((by norm_num))'),
    (['2 * Real.pi', '-Real.pi', '1', '(by positivity)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (T a φ : ℝ) (hT : 0 < T), ∃! ψ, ψ ∈ Set.Ico a (a + T) ∧ ∃ k : ℤ, ψ = φ + k * T := by\n  intro T a φ hT\n  refine ⟨toIcoMod hT a φ, ⟨toIcoMod_mem_Ico hT a φ, ⟨-toIcoDiv hT a φ, ?_⟩⟩, ?_⟩\n  · rw [toIcoMod]; push_cast; rw [zsmul_eq_mul]; ring\n  · rintro ψ ⟨hψ, k, rfl⟩\n    exact ((toIcoMod_eq_iff hT).2 ⟨hψ, -k, by rw [zsmul_eq_mul]; push_cast; ring⟩).symm\nexact @key (2 * Real.pi) (-Real.pi) (1) ((by positivity))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
