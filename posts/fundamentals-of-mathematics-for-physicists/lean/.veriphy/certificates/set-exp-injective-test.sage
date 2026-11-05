# name: set-exp-injective-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.exp_mul_injective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : a ≠ 0), Function.Injective (fun x => Real.exp (a * x)) := by\n  intro a ha\n  exact (Real.exp_injective.comp (mul_right_injective₀ ha))\nexact @key (1) ((by norm_num))'),
    (['2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : a ≠ 0), Function.Injective (fun x => Real.exp (a * x)) := by\n  intro a ha\n  exact (Real.exp_injective.comp (mul_right_injective₀ ha))\nexact @key (2) ((by norm_num))'),
    (['-1', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a : ℝ) (ha : a ≠ 0), Function.Injective (fun x => Real.exp (a * x)) := by\n  intro a ha\n  exact (Real.exp_injective.comp (mul_right_injective₀ ha))\nexact @key (-1) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
