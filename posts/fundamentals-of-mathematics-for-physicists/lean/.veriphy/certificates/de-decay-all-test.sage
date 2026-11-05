# name: de-decay-all-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.decay_all_solutions file "de-decay-all-test.sage"
# description: Sage's desolve of y' = -ky with y(0) = y₀ gives y₀e^{-kx}: k = 0 (edge), 2, -1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', 'fun _ => 3', '_', '5',
                    lhs=desolve(diff(function('Y')(x),x) == 0, function('Y')(x), ics=[0,3]).subs(x=5),
                    rhs=3*exp(0), relation='==', edge=True),
    oracle_instance('2', 'fun x => Real.exp (-2 * x)', '_', '1',
                    lhs=desolve(diff(function('Y')(x),x) == -2*function('Y')(x), function('Y')(x), ics=[0,1]).subs(x=1),
                    rhs=exp(-2), relation='==', edge=False),
    oracle_instance('-1', 'fun x => 4 * Real.exp x', '_', '2',
                    lhs=desolve(diff(function('Y')(x),x) == function('Y')(x), function('Y')(x), ics=[0,4]).subs(x=2),
                    rhs=4*exp(2), relation='==', edge=False),
]
