# name: de-separable-constant-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.separable_constant file "de-separable-constant-test.sage"
# description: the constant y⋆ at a zero of g satisfies y' = f g(y⋆) = 0: g = y(1 - y) at y⋆ = 0 (edge), 1; g = cos at π/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun _ => 1', 'fun y => y * (1 - y)', '0', '_', '1',
                    lhs=diff(SR(0), x),
                    rhs=1*0*(1-0), relation='==', edge=True),
    oracle_instance('fun x => x', 'fun y => y * (1 - y)', '1', '_', '2',
                    lhs=diff(SR(1), x),
                    rhs=2*1*(1-1), relation='==', edge=False),
    oracle_instance('Real.exp', 'Real.cos', 'Real.pi / 2', '_', '0',
                    lhs=diff(pi/2, x),
                    rhs=exp(0)*cos(pi/2), relation='==', edge=False),
]
