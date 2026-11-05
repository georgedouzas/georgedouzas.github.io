# name: de-circles-solve-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ODEComputations.circles_solve file "de-circles-solve-test.sage"
# description: on x² + y² = C, y' = -x/y: upper half-circles of C = 1, 4, 25 at x = 0 (edge), 1, 3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', 'fun s => Real.sqrt (1 - s ^ 2)', '0', '0', '_', '_', '_',
                    lhs=diff(sqrt(1-x^2), x).subs(x=0),
                    rhs=-0/sqrt(1), relation='==', edge=True),
    oracle_instance('4', 'fun s => Real.sqrt (4 - s ^ 2)', '_', '1', '_', '_', '_',
                    lhs=diff(sqrt(4-x^2), x).subs(x=1),
                    rhs=-1/sqrt(3), relation='==', edge=False),
    oracle_instance('25', 'fun s => Real.sqrt (25 - s ^ 2)', '_', '3', '_', '_', '_',
                    lhs=diff(sqrt(25-x^2), x).subs(x=3),
                    rhs=-3/4, relation='==', edge=False),
]
