# name: de-secular-unbounded-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Resonance.secular_unbounded file "de-secular-unbounded-test.sage"
# description: the witness t = (2k + ½)π/ω₀ exceeds M: (F, ω₀, M) = (1, 1, 0) (edge), (2, 3, 100), (-1, 1/2, 10)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '_', '_', '0',
                    lhs=RIF(abs(1/2*(pi/2)*sin(pi/2))),
                    rhs=RIF(0), relation='>', edge=True),
    oracle_instance('2', '3', '_', '_', '100',
                    lhs=RIF(abs(2/6*((2*1000+1/2)*pi/3)*sin(3*((2*1000+1/2)*pi/3)))),
                    rhs=RIF(100), relation='>', edge=False),
    oracle_instance('-1', '1/2', '_', '_', '10',
                    lhs=RIF(abs(-1*((2*100+1/2)*pi*2)*sin((2*100+1/2)*pi))),
                    rhs=RIF(10), relation='>', edge=False),
]
