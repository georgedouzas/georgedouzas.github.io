# name: de-gaussian-limits-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.gaussian_limits file "de-gaussian-limits-test.sage"
# description: Sage's limit of e^{cε²} at ε → 0 is 1 for c = 0 (edge), -1/4, 1/4
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=limit(exp((0)*x^2), x=0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('-1/4',
                    lhs=limit(exp((-1/4)*x^2), x=0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/4',
                    lhs=limit(exp((1/4)*x^2), x=0),
                    rhs=1, relation='==', edge=False),
]
