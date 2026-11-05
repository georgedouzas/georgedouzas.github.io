# name: de-pendulum-equilibria-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.pendulum_equilibria file "de-pendulum-equilibria-test.sage"
# description: -sin(kπ) - γ·0 = 0 at k = 0 (edge), 3, -2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=-sin((0)*pi) - (0)*0,
                    rhs=0, relation='==', edge=True),
    oracle_instance('1/2', '3',
                    lhs=-sin((3)*pi) - (1/2)*0,
                    rhs=0, relation='==', edge=False),
    oracle_instance('2', '-2',
                    lhs=-sin((-2)*pi) - (2)*0,
                    rhs=0, relation='==', edge=False),
]
