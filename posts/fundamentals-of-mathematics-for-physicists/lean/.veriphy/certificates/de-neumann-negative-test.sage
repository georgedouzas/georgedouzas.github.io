# name: de-neumann-negative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleProblems.neumann_negative file "de-neumann-negative-test.sage"
# description: the Neumann conditions with λ = -μ² force C₁ = C₂ = 0: Sage's solve at μ = 1, 2, 1/3 (the zero solution, edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
C1, C2 = var('C1 C2')
instances = [
    oracle_instance('1', '0', '0', '_', '_', '_',
                    lhs=len(solve([(1)*(C1 - C2) == 0, (1)*(C1*exp((1)*pi) - C2*exp(-(1)*pi)) == 0], C1, C2)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '0', '0', '_', '_', '_',
                    lhs=len(solve([(2)*(C1 - C2) == 0, (2)*(C1*exp((2)*pi) - C2*exp(-(2)*pi)) == 0], C1, C2)),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/3', '0', '0', '_', '_', '_',
                    lhs=len(solve([(1/3)*(C1 - C2) == 0, (1/3)*(C1*exp((1/3)*pi) - C2*exp(-(1/3)*pi)) == 0], C1, C2)),
                    rhs=1, relation='==', edge=False),
]
