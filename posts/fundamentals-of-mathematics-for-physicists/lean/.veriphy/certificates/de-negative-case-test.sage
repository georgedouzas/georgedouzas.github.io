# name: de-negative-case-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.negative_case_trivial file "de-negative-case-test.sage"
# description: C(e^{μπ} - e^{-μπ}) = 0 forces C = 0: Sage's solve at μ = 1, 1/10 and C = 0 (edge) itself
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
C_ = var('C_')
instances = [
    oracle_instance('1', '0', '_', '_',
                    lhs=solve(C_*(exp(pi) - exp(-pi)) == 0, C_)[0].rhs(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1/10', '0', '_', '_',
                    lhs=solve(C_*(exp(pi/10) - exp(-pi/10)) == 0, C_)[0].rhs(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('3', '0', '_', '_',
                    lhs=solve(C_*(exp(3*pi) - exp(-3*pi)) == 0, C_)[0].rhs(),
                    rhs=0, relation='==', edge=False),
]
