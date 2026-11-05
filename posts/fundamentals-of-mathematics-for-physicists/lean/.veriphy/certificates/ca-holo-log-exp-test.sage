# name: ca-holo-log-exp-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_exp_strip
# description: Log e^z = z at z = 0, i, 1 + 2i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by simp; linarith [Real.pi_pos])', '(by simp; linarith [Real.pi_pos])'], 'exact Complex.log_exp (by simp; linarith [Real.pi_pos]) (by simp; linarith [Real.pi_pos])', True),
    (['Complex.I', '(by simp; linarith [Real.pi_gt_three])', '(by simp; linarith [Real.pi_gt_three])'], 'exact Complex.log_exp (by simp; linarith [Real.pi_gt_three]) (by simp; linarith [Real.pi_gt_three])', False),
    (['(1 + 2 * Complex.I)', '(by simp; linarith [Real.pi_gt_three])', '(by simp; linarith [Real.pi_gt_three])'], 'exact Complex.log_exp (by simp; linarith [Real.pi_gt_three]) (by simp; linarith [Real.pi_gt_three])', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
