# name: ca-num-no-sqrt-branch-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.no_sqrt_branch file "ca-num-no-sqrt-branch-test.sage"
# description: every candidate continuous function on the punctured plane fails to be a square root; instances are functions with an explicit failing point; instances with an edge case first
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
instances.append(kernel_instance(*['(fun _ => 1)', 'continuousOn_const'], proof='intro h\nhave h1 := h 2 two_ne_zero\nbeta_reduce at h1\nnorm_num at h1', edge=True))
instances.append(kernel_instance(*['(fun z => z)', 'continuousOn_id'], proof='intro h\nhave h1 := h 2 two_ne_zero\nbeta_reduce at h1\nnorm_num at h1', edge=False))
instances.append(kernel_instance(*['(fun z => -z)', '(by fun_prop)'], proof='intro h\nhave h1 := h 2 two_ne_zero\nbeta_reduce at h1\nnorm_num at h1', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
