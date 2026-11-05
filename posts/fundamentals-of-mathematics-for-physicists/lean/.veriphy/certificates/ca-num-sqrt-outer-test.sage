# name: ca-num-sqrt-outer-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.sqrt_branch_outer_cuts file "ca-num-sqrt-outer-test.sage"
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
instances.append(kernel_instance(*['0'], proof='refine ⟨fun z hz => ?_, fun z _ => ?_⟩\n· exact (continuousAt_const.mul ((continuousAt_cpow_const hz).comp (f := fun w : ℂ => 0 - w ^ 2) (x := z)\n    (by fun_prop))).continuousWithinAt\n· show (I * (_ - z ^ 2) ^ (1 / 2 : ℂ)) ^ 2 = z ^ 2 - _\n  rw [mul_pow, one_div, Complex.cpow_ofNat_inv_pow, I_sq]\n  ring', edge=True))
instances.append(kernel_instance(*['1'], proof='refine ⟨fun z hz => ?_, fun z _ => ?_⟩\n· exact (continuousAt_const.mul ((continuousAt_cpow_const hz).comp (f := fun w : ℂ => 1 - w ^ 2) (x := z)\n    (by fun_prop))).continuousWithinAt\n· show (I * (_ - z ^ 2) ^ (1 / 2 : ℂ)) ^ 2 = z ^ 2 - _\n  rw [mul_pow, one_div, Complex.cpow_ofNat_inv_pow, I_sq]\n  ring', edge=False))
instances.append(kernel_instance(*['Complex.I'], proof='refine ⟨fun z hz => ?_, fun z _ => ?_⟩\n· exact (continuousAt_const.mul ((continuousAt_cpow_const hz).comp (f := fun w : ℂ => Complex.I - w ^ 2) (x := z)\n    (by fun_prop))).continuousWithinAt\n· show (I * (_ - z ^ 2) ^ (1 / 2 : ℂ)) ^ 2 = z ^ 2 - _\n  rw [mul_pow, one_div, Complex.cpow_ofNat_inv_pow, I_sq]\n  ring', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
