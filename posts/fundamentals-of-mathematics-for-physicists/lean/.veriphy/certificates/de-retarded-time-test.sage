# name: de-retarded-time-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.RetardedGreens.retarded_time_derivative file "de-retarded-time-test.sage"
# description: d/dτ(τ + |x - y₀(τ)|) = 1 - v by Sage for y₀ = vτ, τ², 0 (edge), at points with y₀ < x
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('5', '_', '1/2', '1', '_', '_',
                    lhs=diff(x + abs(5 - x/2), x).subs(x=1),
                    rhs=1 - 1/2, relation='==', edge=False),
    oracle_instance('3', '_', '2', '1', '_', '_',
                    lhs=diff(x + abs(3 - x^2), x).subs(x=1),
                    rhs=1 - 2, relation='==', edge=False),
    oracle_instance('1', 'fun _ => 0', '0', '0', '_', '_',
                    lhs=diff(x + abs(1 - 0), x),
                    rhs=1, relation='==', edge=True),
]
