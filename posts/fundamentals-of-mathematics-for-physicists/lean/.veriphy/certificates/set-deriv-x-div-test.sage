# name: set-deriv-x-div-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Homeomorphism.deriv_x_div
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ) (hx : 1 - x ^ 2 ≠ 0), deriv (fun x : ℝ => x / (1 - x ^ 2)) x = (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by\n  intro x hx\n  have h := (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_pow 2 x)) hx\n  refine h.deriv.trans ?_\n  simp only [Pi.sub_apply, id]\n  field_simp\n  ring\nexact @key (0) ((by norm_num))'),
    (['1/2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ) (hx : 1 - x ^ 2 ≠ 0), deriv (fun x : ℝ => x / (1 - x ^ 2)) x = (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by\n  intro x hx\n  have h := (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_pow 2 x)) hx\n  refine h.deriv.trans ?_\n  simp only [Pi.sub_apply, id]\n  field_simp\n  ring\nexact @key (1/2) ((by norm_num))'),
    (['2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ) (hx : 1 - x ^ 2 ≠ 0), deriv (fun x : ℝ => x / (1 - x ^ 2)) x = (1 + x ^ 2) / (1 - x ^ 2) ^ 2 := by\n  intro x hx\n  have h := (hasDerivAt_id x).div ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_pow 2 x)) hx\n  refine h.deriv.trans ?_\n  simp only [Pi.sub_apply, id]\n  field_simp\n  ring\nexact @key (2) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
