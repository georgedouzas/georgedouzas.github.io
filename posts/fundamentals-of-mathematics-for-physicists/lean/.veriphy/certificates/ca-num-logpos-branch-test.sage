# name: ca-num-logpos-branch-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.logPos_branch file "ca-num-logpos-branch-test.sage"
# description: a continuity and branch statement; each instance is rechecked by the kernel through the general argument; instances with an edge case first
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
instances.append(kernel_instance(*['0'], proof='refine ⟨fun z hz => ?_, fun z hz => ?_⟩\n· exact (((continuousAt_clog hz).comp continuous_neg.continuousAt |>.add continuousAt_const).add\n    continuousAt_const).continuousWithinAt\n· show Complex.exp (Complex.log (-z) + π * I + 2 * π * I * _) = z\n  rw [Complex.exp_add, Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), Complex.exp_pi_mul_I,\n    mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I]\n  ring', edge=True))
instances.append(kernel_instance(*['1'], proof='refine ⟨fun z hz => ?_, fun z hz => ?_⟩\n· exact (((continuousAt_clog hz).comp continuous_neg.continuousAt |>.add continuousAt_const).add\n    continuousAt_const).continuousWithinAt\n· show Complex.exp (Complex.log (-z) + π * I + 2 * π * I * _) = z\n  rw [Complex.exp_add, Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), Complex.exp_pi_mul_I,\n    mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I]\n  ring', edge=False))
instances.append(kernel_instance(*['(-1)'], proof='refine ⟨fun z hz => ?_, fun z hz => ?_⟩\n· exact (((continuousAt_clog hz).comp continuous_neg.continuousAt |>.add continuousAt_const).add\n    continuousAt_const).continuousWithinAt\n· show Complex.exp (Complex.log (-z) + π * I + 2 * π * I * _) = z\n  rw [Complex.exp_add, Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), Complex.exp_pi_mul_I,\n    mul_comm (2 * π * I) _, Complex.exp_int_mul_two_pi_mul_I]\n  ring', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
