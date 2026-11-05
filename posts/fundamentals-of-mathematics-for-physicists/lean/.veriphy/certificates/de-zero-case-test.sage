# name: de-zero-case-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BoundaryValueProblems.zero_case_trivial file "de-zero-case-test.sage"
# description: C₁ + C₂x vanishing at 0 and π is zero: Sage's solve of the two conditions (and the trivial solution, edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
C1, C2 = var('C1 C2')
instances = [
    oracle_instance('0', '0', '_', '_',
                    lhs=len(solve([C1 + C2*0 == 0, C1 + C2*pi == 0], C1, C2)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('0', '0', '_', '_',
                    lhs=solve([C1 + C2*0 == 0, C1 + C2*pi == 0], C1, C2)[0][0].rhs(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('0', '0', '_', '_',
                    lhs=solve([C1 + C2*0 == 0, C1 + C2*pi == 0], C1, C2)[0][1].rhs(),
                    rhs=0, relation='==', edge=False),
]
