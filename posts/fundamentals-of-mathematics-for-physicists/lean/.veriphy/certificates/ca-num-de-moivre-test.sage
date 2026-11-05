# name: ca-num-de-moivre-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.de_moivre file "ca-num-de-moivre-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
CI = ComplexIntervalField(128)
def cdiff(a, b):
    d = CI(a) - CI(b)
    return RI(d.real().abs() + d.imag().abs())
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(kernel_instance(*['3', '0'], proof='simp', edge=False))
instances.append(oracle_instance(*['2', 'Real.pi'], lhs=cdiff((cos(pi)+I*sin(pi))^2, cos(2*pi)+I*sin(2*pi)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['3', '(1 / 2)'], lhs=cdiff((cos(1/2)+I*sin(1/2))^3, cos(3/2)+I*sin(3/2)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
