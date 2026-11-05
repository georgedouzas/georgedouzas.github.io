# name: ca-holo-log-inverse-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicWorked.log_deriv_by_inverse
# description: Log' w = 1/w by the inverse rule at w = 1, i, 2 + i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['1', '(by norm_num [Complex.mem_slitPlane_iff])'], 'have hw := (by norm_num [Complex.mem_slitPlane_iff] : (1 : ℂ) ∈ Complex.slitPlane)\nhave h := inverse_deriv Complex.exp Complex.log (Complex.exp (Complex.log (1))) (1) (_root_.continuousAt_clog hw)\n  (Complex.hasDerivAt_exp _) (Complex.exp_ne_zero _)\n  (Filter.eventually_of_mem (isOpen_ne.mem_nhds (slitPlane_ne_zero hw)) fun y hy => Complex.exp_log hy)\nrwa [Complex.exp_log (slitPlane_ne_zero hw)] at h', True),
    (['Complex.I', '(by norm_num [Complex.mem_slitPlane_iff])'], 'have hw := (by norm_num [Complex.mem_slitPlane_iff] : (Complex.I : ℂ) ∈ Complex.slitPlane)\nhave h := inverse_deriv Complex.exp Complex.log (Complex.exp (Complex.log (Complex.I))) (Complex.I) (_root_.continuousAt_clog hw)\n  (Complex.hasDerivAt_exp _) (Complex.exp_ne_zero _)\n  (Filter.eventually_of_mem (isOpen_ne.mem_nhds (slitPlane_ne_zero hw)) fun y hy => Complex.exp_log hy)\nrwa [Complex.exp_log (slitPlane_ne_zero hw)] at h', False),
    (['(2 + Complex.I)', '(by norm_num [Complex.mem_slitPlane_iff])'], 'have hw := (by norm_num [Complex.mem_slitPlane_iff] : (2 + Complex.I : ℂ) ∈ Complex.slitPlane)\nhave h := inverse_deriv Complex.exp Complex.log (Complex.exp (Complex.log (2 + Complex.I))) (2 + Complex.I) (_root_.continuousAt_clog hw)\n  (Complex.hasDerivAt_exp _) (Complex.exp_ne_zero _)\n  (Filter.eventually_of_mem (isOpen_ne.mem_nhds (slitPlane_ne_zero hw)) fun y hy => Complex.exp_log hy)\nrwa [Complex.exp_log (slitPlane_ne_zero hw)] at h', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
