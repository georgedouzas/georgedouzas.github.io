# name: ca-ex-reflection-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.Exercises.reflection file "ca-ex-reflection-test.sage"
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
instances.append(oracle_instance(*['(1 / 2)', '(by intro n h; have h2 := congrArg Complex.re h; simp at h2; have h3 : (2 * n : ℤ) = 1 := by exact_mod_cast (by linarith : (2 : ℝ) * n = 1); omega)'], lhs=bdiff(CB(1/2).gamma()^2*(CB(pi)/2).sin(), CB(pi)), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['(1 / 3)', '(by intro n h; have h2 := congrArg Complex.re h; simp at h2; have h3 : (3 * n : ℤ) = 1 := by exact_mod_cast (by linarith : (3 : ℝ) * n = 1); omega)'], lhs=bdiff(CB(1/3).gamma()*CB(2/3).gamma()*(CB(pi)/3).sin(), CB(pi)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(1 / 2 + 2 * Complex.I)', '(by intro n h; have h2 := congrArg Complex.im h; simp at h2)'], lhs=bdiff(CB(1/2+2*I).gamma()*CB(1/2-2*I).gamma()*(CB(pi)*CB(1/2+2*I)).sin(), CB(pi)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
