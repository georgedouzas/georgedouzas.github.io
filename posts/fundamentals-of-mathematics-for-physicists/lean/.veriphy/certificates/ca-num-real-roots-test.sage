# name: ca-num-real-roots-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.NumbersWorked.real_roots file "ca-num-real-roots-test.sage"
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
instances.append(kernel_instance(*['0', '(by norm_num)'], proof='simp', edge=True))
instances.append(kernel_instance(*['3', '(by norm_num)'], proof='rw [ω_pow_values.2.1]\nsimp', edge=False))
instances.append(kernel_instance(*['1', '(by norm_num)'], proof="rw [pow_one, ω_eq]\nsimp [(Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 3)).ne']", edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
