# name: de-string-coeff1-left-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Exercises.string_coeff1_left file "de-string-coeff1-left-test.sage"
# description: ∫₀^ξ x(π-ξ)/π sin(1x) dx by Sage's integrate against the closed form, exactly, at ξ = 0 (edge), 1, π/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0',
                    lhs=(integrate(xs*(pi-(0))/pi*sin(1*xs), xs, 0, 0) - (pi-(0))/pi*(sin(1*(0))/1^2 - (0)*cos(1*(0))/1)).subs(xs=xs).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1',
                    lhs=(integrate(xs*(pi-(1))/pi*sin(1*xs), xs, 0, 1) - (pi-(1))/pi*(sin(1*(1))/1^2 - (1)*cos(1*(1))/1)).subs(xs=xs).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('Real.pi / 2',
                    lhs=(integrate(xs*(pi-(pi/2))/pi*sin(1*xs), xs, 0, pi/2) - (pi-(pi/2))/pi*(sin(1*(pi/2))/1^2 - (pi/2)*cos(1*(pi/2))/1)).subs(xs=xs).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
