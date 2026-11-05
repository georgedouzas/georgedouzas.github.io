# name: ca-num-logpos-im-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.logPos_im file "ca-num-logpos-im-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
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
instances.append(kernel_instance(*['(-1)', '(by simp [Complex.mem_slitPlane_iff])'], proof='simp [logPos, Real.pi_pos]', edge=True))
instances.append(kernel_instance(*['Complex.I', '(by simp [Complex.mem_slitPlane_iff])'], proof='simp [logPos, Complex.log_neg_I]\nconstructor <;> linarith [Real.pi_pos]', edge=False))
instances.append(kernel_instance(*['(-Complex.I)', '(by simp [Complex.mem_slitPlane_iff])'], proof='simp [logPos, Complex.log_I]\nconstructor <;> linarith [Real.pi_pos]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
