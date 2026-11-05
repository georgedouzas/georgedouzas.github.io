# name: ca-num-log-branches-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.log_branches file "ca-num-log-branches-test.sage"
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
instances.append(kernel_instance(*['0'], proof='refine ⟨fun z hz => ((continuousAt_clog hz).add continuousAt_const).continuousWithinAt, fun z hz => ?_⟩\nshow Complex.exp _ = z\nrw [Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), mul_comm (2 * π * I) _,\n  Complex.exp_int_mul_two_pi_mul_I, mul_one]', edge=True))
instances.append(kernel_instance(*['1'], proof='refine ⟨fun z hz => ((continuousAt_clog hz).add continuousAt_const).continuousWithinAt, fun z hz => ?_⟩\nshow Complex.exp _ = z\nrw [Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), mul_comm (2 * π * I) _,\n  Complex.exp_int_mul_two_pi_mul_I, mul_one]', edge=False))
instances.append(kernel_instance(*['(-2)'], proof='refine ⟨fun z hz => ((continuousAt_clog hz).add continuousAt_const).continuousWithinAt, fun z hz => ?_⟩\nshow Complex.exp _ = z\nrw [Complex.exp_add, Complex.exp_log (slitPlane_ne_zero hz), mul_comm (2 * π * I) _,\n  Complex.exp_int_mul_two_pi_mul_I, mul_one]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
