# name: ca-num-quadratic-irred-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.NumbersWorked.quadratic_irreducible file "ca-num-quadratic-irred-test.sage"
# description: irreducibility of a family of quadratics; each instance is rechecked by the kernel through the root test; instances with an edge case first
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
instances.append(kernel_instance(*['0', '1', '(by norm_num)'], proof='apply Polynomial.irreducible_of_degree_le_three_of_not_isRoot\n· rw [show (Polynomial.X ^ 2 + Polynomial.C _ * Polynomial.X + Polynomial.C _ : Polynomial ℚ).natDegree = 2 by compute_degree!]\n  decide\n· intro x hx\n  simp at hx\n  nlinarith [sq_nonneg (2 * x + 0)]', edge=True))
instances.append(kernel_instance(*['1', '1', '(by norm_num)'], proof='apply Polynomial.irreducible_of_degree_le_three_of_not_isRoot\n· rw [show (Polynomial.X ^ 2 + Polynomial.C _ * Polynomial.X + Polynomial.C _ : Polynomial ℚ).natDegree = 2 by compute_degree!]\n  decide\n· intro x hx\n  simp at hx\n  nlinarith [sq_nonneg (2 * x + 1)]', edge=False))
instances.append(kernel_instance(*['(-1)', '1', '(by norm_num)'], proof='apply Polynomial.irreducible_of_degree_le_three_of_not_isRoot\n· rw [show (Polynomial.X ^ 2 + Polynomial.C _ * Polynomial.X + Polynomial.C _ : Polynomial ℚ).natDegree = 2 by compute_degree!]\n  decide\n· intro x hx\n  simp at hx\n  nlinarith [sq_nonneg (2 * x + (-1))]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
