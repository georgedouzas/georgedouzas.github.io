# name: de-beats-limit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Resonance.beats_limit file "de-beats-limit-test.sage"
# description: Sage's limit ω → ω₀ of the beats solution equals (F/2ω₀)t sin ω₀t at (F, ω₀, t) = (1, 1, 0) (edge), (2, 3, 1), (1, 1/2, 5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
w_ = var('w_')
instances = [
    oracle_instance('1', '1', '0', '_',
                    lhs=limit((1)/((1)^2 - w_^2)*(cos(w_*(0)) - cos((1)*(0))), w_=(1)),
                    rhs=(1)/(2*(1))*(0)*sin((1)*(0)), relation='==', edge=True),
    oracle_instance('2', '3', '1', '_',
                    lhs=limit((2)/((3)^2 - w_^2)*(cos(w_*(1)) - cos((3)*(1))), w_=(3)),
                    rhs=(2)/(2*(3))*(1)*sin((3)*(1)), relation='==', edge=False),
    oracle_instance('1', '1/2', '5', '_',
                    lhs=limit((1)/((1/2)^2 - w_^2)*(cos(w_*(5)) - cos((1/2)*(5))), w_=(1/2)),
                    rhs=(1)/(2*(1/2))*(5)*sin((1/2)*(5)), relation='==', edge=False),
]
