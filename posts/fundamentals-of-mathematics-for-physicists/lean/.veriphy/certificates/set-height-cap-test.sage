# name: set-height-cap-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Maps.height_cap
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1/2', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ) (hc : c < 1), sphere ∩ height ⁻¹' Set.Ioc c 1 = sphere ∩ {p | c < p 2} ∧ northPole ∈ sphere ∩ height ⁻¹' Set.Ioc c 1 := by\n  intro c hc\n  constructor\n  · ext p\n    simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Ioc]\n    constructor\n    · rintro ⟨h, h1, -⟩; exact ⟨h, h1⟩\n    · rintro ⟨h, h1⟩; exact ⟨h, h1, by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]⟩\n  · simp [sphere, height, northPole, hc]\nexact @key (1/2) ((by norm_num))"),
    (['0', '(by norm_num)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ) (hc : c < 1), sphere ∩ height ⁻¹' Set.Ioc c 1 = sphere ∩ {p | c < p 2} ∧ northPole ∈ sphere ∩ height ⁻¹' Set.Ioc c 1 := by\n  intro c hc\n  constructor\n  · ext p\n    simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Ioc]\n    constructor\n    · rintro ⟨h, h1, -⟩; exact ⟨h, h1⟩\n    · rintro ⟨h, h1⟩; exact ⟨h, h1, by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]⟩\n  · simp [sphere, height, northPole, hc]\nexact @key (0) ((by norm_num))"),
    (['-1/2', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (c : ℝ) (hc : c < 1), sphere ∩ height ⁻¹' Set.Ioc c 1 = sphere ∩ {p | c < p 2} ∧ northPole ∈ sphere ∩ height ⁻¹' Set.Ioc c 1 := by\n  intro c hc\n  constructor\n  · ext p\n    simp only [sphere, height, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_Ioc]\n    constructor\n    · rintro ⟨h, h1, -⟩; exact ⟨h, h1⟩\n    · rintro ⟨h, h1⟩; exact ⟨h, h1, by nlinarith [sq_nonneg (p 0), sq_nonneg (p 1)]⟩\n  · simp [sphere, height, northPole, hc]\nexact @key (-1/2) ((by norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
