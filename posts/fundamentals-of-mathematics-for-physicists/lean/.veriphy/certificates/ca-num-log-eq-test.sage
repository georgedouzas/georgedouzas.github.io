# name: ca-num-log-eq-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_eq file "ca-num-log-eq-test.sage"
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
instances.append(kernel_instance(*['0'], proof='refine ⟨?_, ?_, ?_⟩\n· apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]\n· exact Complex.neg_pi_lt_log_im _\n· exact Complex.log_im_le_pi _', edge=True))
instances.append(kernel_instance(*['1'], proof='refine ⟨?_, ?_, ?_⟩\n· apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]\n· exact Complex.neg_pi_lt_log_im _\n· exact Complex.log_im_le_pi _', edge=False))
instances.append(kernel_instance(*['Complex.I'], proof='refine ⟨?_, ?_, ?_⟩\n· apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]\n· exact Complex.neg_pi_lt_log_im _\n· exact Complex.log_im_le_pi _', edge=False))
instances.append(kernel_instance(*['(-1)'], proof='refine ⟨?_, ?_, ?_⟩\n· apply Complex.ext <;> simp [Complex.log_re, Complex.log_im]\n· exact Complex.neg_pi_lt_log_im _\n· exact Complex.log_im_le_pi _', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
