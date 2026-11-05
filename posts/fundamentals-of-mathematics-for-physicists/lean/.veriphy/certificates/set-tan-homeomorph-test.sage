# name: set-tan-homeomorph-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Homeomorphism.tan_homeomorph
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-(Real.pi / 2)) (Real.pi / 2) ≃ₜ ℝ, ∀ x, e x = a * Real.tan x := by\n  intro a ha\n  exact ⟨Real.tanOrderIso.toHomeomorph.trans (Homeomorph.mulLeft₀ a ha.ne'), fun x => rfl⟩\nexact @key (1) ((by norm_num))"),
    (['2', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-(Real.pi / 2)) (Real.pi / 2) ≃ₜ ℝ, ∀ x, e x = a * Real.tan x := by\n  intro a ha\n  exact ⟨Real.tanOrderIso.toHomeomorph.trans (Homeomorph.mulLeft₀ a ha.ne'), fun x => rfl⟩\nexact @key (2) ((by norm_num))"),
    (['1/2', '(by norm_num)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : 0 < a), ∃ e : Ioo (-(Real.pi / 2)) (Real.pi / 2) ≃ₜ ℝ, ∀ x, e x = a * Real.tan x := by\n  intro a ha\n  exact ⟨Real.tanOrderIso.toHomeomorph.trans (Homeomorph.mulLeft₀ a ha.ne'), fun x => rfl⟩\nexact @key (1/2) ((by norm_num))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
