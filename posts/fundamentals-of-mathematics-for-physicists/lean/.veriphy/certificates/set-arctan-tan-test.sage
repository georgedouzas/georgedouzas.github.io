# name: set-arctan-tan-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Inverse.arctan_tan_inverse
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '0', '(by constructor <;> linarith [Real.pi_pos])'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x y : ℝ) (hx : x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)), Real.arctan (Real.tan x) = x ∧ Real.tan (Real.arctan y) = y := by\n  intro x y hx\n  exact (⟨Real.arctan_tan hx.1 hx.2, Real.tan_arctan y⟩)\nexact @key (0) (0) ((by constructor <;> linarith [Real.pi_pos]))'),
    (['1', '5', '(by constructor <;> linarith [Real.pi_gt_three])'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x y : ℝ) (hx : x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)), Real.arctan (Real.tan x) = x ∧ Real.tan (Real.arctan y) = y := by\n  intro x y hx\n  exact (⟨Real.arctan_tan hx.1 hx.2, Real.tan_arctan y⟩)\nexact @key (1) (5) ((by constructor <;> linarith [Real.pi_gt_three]))'),
    (['-1', '-2', '(by constructor <;> linarith [Real.pi_gt_three])'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x y : ℝ) (hx : x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)), Real.arctan (Real.tan x) = x ∧ Real.tan (Real.arctan y) = y := by\n  intro x y hx\n  exact (⟨Real.arctan_tan hx.1 hx.2, Real.tan_arctan y⟩)\nexact @key (-1) (-2) ((by constructor <;> linarith [Real.pi_gt_three]))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
