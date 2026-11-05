# name: de-picard-unique-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.picard_unique file "de-picard-unique-test.sage"
# description: two solutions of a Lipschitz equation agreeing at x₀ coincide: Sage's desolve with the initial value returns one solution, equal to the given ones, for y' = y, y' = -2y + 1, y' = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun _ y => y', '1', '_', 'Real.exp', 'Real.exp', '_', '_', '0', '_',
                    lhs=desolve(diff(function('Y')(x), x) == function('Y')(x), function('Y')(x), ics=[0, 1]),
                    rhs=exp(x), relation='==', edge=False),
    oracle_instance('fun _ y => -2 * y + 1', '2', '_', 'fun x => 1/2 + Real.exp (-2 * x)', 'fun x => 1/2 + Real.exp (-2 * x)', '_', '_', '0', '_',
                    lhs=desolve(diff(function('Y')(x), x) == -2*function('Y')(x) + 1, function('Y')(x), ics=[0, 3/2]).expand(),
                    rhs=(1/2 + exp(-2*x)).expand(), relation='==', edge=False),
    oracle_instance('fun _ _ => 0', '0', '_', 'fun _ => 5', 'fun _ => 5', '_', '_', '1', '_',
                    lhs=desolve(diff(function('Y')(x), x) == 0, function('Y')(x), ics=[1, 5]),
                    rhs=5, relation='==', edge=True),
]
