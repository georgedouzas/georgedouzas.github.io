# name: set-affine-solution-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Bijective.affine_unique_solution
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2', '1', '5', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b y : ℝ) (ha : a ≠ 0), ∃! x, a * x + b = y ∧ x = (y - b) / a := by\n  intro a b y ha\n  refine ⟨(y - b) / a, ⟨by field_simp; ring, rfl⟩, fun x hx => hx.2⟩\nexact @key (2) (1) (5) ((by norm_num))'),
    (['1', '0', '0', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b y : ℝ) (ha : a ≠ 0), ∃! x, a * x + b = y ∧ x = (y - b) / a := by\n  intro a b y ha\n  refine ⟨(y - b) / a, ⟨by field_simp; ring, rfl⟩, fun x hx => hx.2⟩\nexact @key (1) (0) (0) ((by norm_num))'),
    (['-3', '2', '7', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b y : ℝ) (ha : a ≠ 0), ∃! x, a * x + b = y ∧ x = (y - b) / a := by\n  intro a b y ha\n  refine ⟨(y - b) / a, ⟨by field_simp; ring, rfl⟩, fun x hx => hx.2⟩\nexact @key (-3) (2) (7) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
