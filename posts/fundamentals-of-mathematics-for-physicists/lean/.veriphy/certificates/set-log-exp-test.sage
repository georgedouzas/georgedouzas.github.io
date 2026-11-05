# name: set-log-exp-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Inverse.log_exp_inverse
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x y : ℝ) (hy : 0 < y), Real.log (Real.exp x) = x ∧ Real.exp (Real.log y) = y := by\n  intro x y hy\n  exact (⟨Real.log_exp x, Real.exp_log hy⟩)\nexact @key (0) (1) ((by norm_num))'),
    (['2', '3', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x y : ℝ) (hy : 0 < y), Real.log (Real.exp x) = x ∧ Real.exp (Real.log y) = y := by\n  intro x y hy\n  exact (⟨Real.log_exp x, Real.exp_log hy⟩)\nexact @key (2) (3) ((by norm_num))'),
    (['-1', '1/2', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x y : ℝ) (hy : 0 < y), Real.log (Real.exp x) = x ∧ Real.exp (Real.log y) = y := by\n  intro x y hy\n  exact (⟨Real.log_exp x, Real.exp_log hy⟩)\nexact @key (-1) (1/2) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
