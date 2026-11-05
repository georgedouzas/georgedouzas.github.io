# name: de-steady-state-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Resonance.steady_state_solves file "de-steady-state-solves-test.sage"
# description: ẍ + 2γẋ + ω₀²x - F cos ωt for the steady-state amplitude and phase, by Sage's derivatives, exactly, at (γ, ω₀, ω, F, t) = (1, 1, 1, 1, 0) (edge), (1/2, 2, 1, 3, 1), (0, 1, 2, 1, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '1', '1', '1', '0', '_',
                    lhs=(diff(((1)*((1)^2-(1)^2)/(((1)^2-(1)^2)^2+4*(1)^2*(1)^2)*cos((1)*s_) + 2*(1)*(1)*(1)/(((1)^2-(1)^2)^2+4*(1)^2*(1)^2)*sin((1)*s_)), s_, 2) + 2*(1)*diff(((1)*((1)^2-(1)^2)/(((1)^2-(1)^2)^2+4*(1)^2*(1)^2)*cos((1)*s_) + 2*(1)*(1)*(1)/(((1)^2-(1)^2)^2+4*(1)^2*(1)^2)*sin((1)*s_)), s_) + (1)^2*((1)*((1)^2-(1)^2)/(((1)^2-(1)^2)^2+4*(1)^2*(1)^2)*cos((1)*s_) + 2*(1)*(1)*(1)/(((1)^2-(1)^2)^2+4*(1)^2*(1)^2)*sin((1)*s_)) - (1)*cos((1)*s_)).subs(s_=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1/2', '2', '1', '3', '1', '_',
                    lhs=(diff(((3)*((2)^2-(1)^2)/(((2)^2-(1)^2)^2+4*(1/2)^2*(1)^2)*cos((1)*s_) + 2*(1/2)*(1)*(3)/(((2)^2-(1)^2)^2+4*(1/2)^2*(1)^2)*sin((1)*s_)), s_, 2) + 2*(1/2)*diff(((3)*((2)^2-(1)^2)/(((2)^2-(1)^2)^2+4*(1/2)^2*(1)^2)*cos((1)*s_) + 2*(1/2)*(1)*(3)/(((2)^2-(1)^2)^2+4*(1/2)^2*(1)^2)*sin((1)*s_)), s_) + (2)^2*((3)*((2)^2-(1)^2)/(((2)^2-(1)^2)^2+4*(1/2)^2*(1)^2)*cos((1)*s_) + 2*(1/2)*(1)*(3)/(((2)^2-(1)^2)^2+4*(1/2)^2*(1)^2)*sin((1)*s_)) - (3)*cos((1)*s_)).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('0', '1', '2', '1', '1/2', '_',
                    lhs=(diff(((1)*((1)^2-(2)^2)/(((1)^2-(2)^2)^2+4*(0)^2*(2)^2)*cos((2)*s_) + 2*(0)*(2)*(1)/(((1)^2-(2)^2)^2+4*(0)^2*(2)^2)*sin((2)*s_)), s_, 2) + 2*(0)*diff(((1)*((1)^2-(2)^2)/(((1)^2-(2)^2)^2+4*(0)^2*(2)^2)*cos((2)*s_) + 2*(0)*(2)*(1)/(((1)^2-(2)^2)^2+4*(0)^2*(2)^2)*sin((2)*s_)), s_) + (1)^2*((1)*((1)^2-(2)^2)/(((1)^2-(2)^2)^2+4*(0)^2*(2)^2)*cos((2)*s_) + 2*(0)*(2)*(1)/(((1)^2-(2)^2)^2+4*(0)^2*(2)^2)*sin((2)*s_)) - (1)*cos((2)*s_)).subs(s_=1/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
