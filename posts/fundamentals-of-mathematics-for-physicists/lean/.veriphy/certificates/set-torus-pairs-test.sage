# name: set-torus-pairs-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.torus_pairs_of_angles
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '1', '(by norm_num)', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r₁ r₂ : ℝ) (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂), Set.range (fun q : ℝ × ℝ => ((r₁ * Real.cos q.1, r₁ * Real.sin q.1), (r₂ * Real.cos q.2, r₂ * Real.sin q.2))) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₁ ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₂ ^ 2} := by\n  intro r₁ r₂ h₁ h₂\n  rw [← range_wrap r₁ h₁, ← range_wrap r₂ h₂, ← Set.range_prodMap]\n  rfl\nexact @key (1) (1) ((by norm_num)) ((by norm_num))'),
    (['2', '1', '(by norm_num)', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r₁ r₂ : ℝ) (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂), Set.range (fun q : ℝ × ℝ => ((r₁ * Real.cos q.1, r₁ * Real.sin q.1), (r₂ * Real.cos q.2, r₂ * Real.sin q.2))) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₁ ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₂ ^ 2} := by\n  intro r₁ r₂ h₁ h₂\n  rw [← range_wrap r₁ h₁, ← range_wrap r₂ h₂, ← Set.range_prodMap]\n  rfl\nexact @key (2) (1) ((by norm_num)) ((by norm_num))'),
    (['0', '3', '(by norm_num)', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r₁ r₂ : ℝ) (h₁ : 0 ≤ r₁) (h₂ : 0 ≤ r₂), Set.range (fun q : ℝ × ℝ => ((r₁ * Real.cos q.1, r₁ * Real.sin q.1), (r₂ * Real.cos q.2, r₂ * Real.sin q.2))) = {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₁ ^ 2} ×ˢ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r₂ ^ 2} := by\n  intro r₁ r₂ h₁ h₂\n  rw [← range_wrap r₁ h₁, ← range_wrap r₂ h₂, ← Set.range_prodMap]\n  rfl\nexact @key (0) (3) ((by norm_num)) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
