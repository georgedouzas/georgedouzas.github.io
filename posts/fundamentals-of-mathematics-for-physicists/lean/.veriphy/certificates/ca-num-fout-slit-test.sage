# name: ca-num-fout-slit-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.fOut_slit file "ca-num-fout-slit-test.sage"
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
instances.append(kernel_instance(*['2', '(by norm_num)'], proof='rw [Complex.mem_slitPlane_iff]\nleft\nnorm_num', edge=True))
instances.append(oracle_instance(*['Complex.I', '(by simp)'], lhs=RI(RI(CI(1-1/I^2).real())), rhs=RI(RI(0)), relation='>', edge=False))
instances.append(oracle_instance(*['(1 + Complex.I)', '(by simp)'], lhs=RI(RI(abs(CI(1-1/(1+I)^2).imag()))), rhs=RI(RI(0)), relation='>', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
