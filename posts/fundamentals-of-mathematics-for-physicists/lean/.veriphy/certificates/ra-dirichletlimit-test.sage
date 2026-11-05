# name: ra-dirichletlimit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or by intervals
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.DirichletLimit file "ra-dirichletlimit-test.sage"
# description: for ε = 1, 1/10, 1/1000 the sine integral Si(R) stays within ε of π/2 for R ≥ R₀ = 4/ε (128-bit intervals at R₀, 2R₀, 10R₀, 100R₀)
SEED = 0  # the instances are listed, no sampling
R = RealIntervalField(128)
def B(b):
    return QQ(1 if bool(b) else 0)
instances = []
for eps in [QQ(1), QQ(1) / 10, QQ(1) / 1000]:
    R0 = 4 / eps
    ok = all(abs(R(sin_integral(R0 * m)) - R(pi) / 2) < R(eps) for m in [1, 2, 10, 100])
    instances.append(oracle_instance(str(eps), '(by norm_num)', lhs=B(ok), rhs=QQ(1), relation='==', edge=bool(eps == 1)))
