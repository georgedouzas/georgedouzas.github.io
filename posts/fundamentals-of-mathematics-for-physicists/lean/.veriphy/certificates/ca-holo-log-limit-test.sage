# name: ca-holo-log-limit-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.log_limit_below
# description: the one-sided limit of Log at x = -1, -2, -1/2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(-1)', '(by norm_num)'], 'exact ⟨Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by norm_num), by rw [Complex.log_im, Complex.arg_ofReal_of_neg (by norm_num)]⟩', True),
    (['(-2)', '(by norm_num)'], 'exact ⟨Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by norm_num), by rw [Complex.log_im, Complex.arg_ofReal_of_neg (by norm_num)]⟩', False),
    (['(-1 / 2)', '(by norm_num)'], 'exact ⟨Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by norm_num), by rw [Complex.log_im, Complex.arg_ofReal_of_neg (by norm_num)]⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
