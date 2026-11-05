# name: ca-num-no-log-branch-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.no_log_branch file "ca-num-no-log-branch-test.sage"
# description: every candidate continuous function on the punctured plane fails to be a logarithm; instances are functions with an explicit failing point; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
CI = ComplexIntervalField(128)
def cdiff(a, b):
    d = CI(a) - CI(b)
    return RI(d.real().abs() + d.imag().abs())
CB = ComplexBallField(128)
def bdiff(a, b):
    d = a - b
    return RI(d.real().abs()) + RI(d.imag().abs())
instances.append(kernel_instance(*['(fun _ => 0)', 'continuousOn_const'], proof='intro h\nhave h1 := h 2 two_ne_zero\nbeta_reduce at h1\nnorm_num at h1', edge=True))
instances.append(kernel_instance(*['(fun z => z)', 'continuousOn_id'], proof='intro h\nhave h1 := h 1 one_ne_zero\nbeta_reduce at h1\nhave h2 : Real.exp 1 = 1 := by exact_mod_cast (show ((Real.exp 1 : ℝ) : ℂ) = 1 by push_cast; exact h1)\nhave := Real.add_one_lt_exp (x := 1) one_ne_zero\nlinarith', edge=False))
instances.append(kernel_instance(*['(fun z => z + 1)', '(by fun_prop)'], proof='intro h\nhave h1 := h (-1) (by norm_num)\nbeta_reduce at h1\nnorm_num at h1', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
