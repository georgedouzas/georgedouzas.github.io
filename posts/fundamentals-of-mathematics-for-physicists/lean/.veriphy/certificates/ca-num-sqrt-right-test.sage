# name: ca-num-sqrt-right-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_right_half file "ca-num-sqrt-right-test.sage"
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
instances.append(kernel_instance(*['1', '(by simp [Complex.mem_slitPlane_iff])'], proof='simp', edge=True))
instances.append(oracle_instance(*['(-1 + Complex.I)', '(by simp [Complex.mem_slitPlane_iff])'], lhs=RI(RI(CI((-1+I)^(1/2)).real())), rhs=RI(RI(0)), relation='>', edge=False))
instances.append(oracle_instance(*['(-4 - Complex.I)', '(by simp [Complex.mem_slitPlane_iff])'], lhs=RI(RI(CI((-4-I)^(1/2)).real())), rhs=RI(RI(0)), relation='>', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
