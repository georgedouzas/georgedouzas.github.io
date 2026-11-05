# name: de-transient-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Resonance.transient_decays file "de-transient-test.sage"
# description: Sage's limit of e^{-γt}f(t) for bounded f: (γ, f) = (1, cos), (1/2, sin 3t), (2, 0) (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '_', 'Real.cos', '_',
                    lhs=limit(exp(-x)*cos(x), x=oo),
                    rhs=0, relation='==', edge=False),
    oracle_instance('1/2', '1', '_', 'fun t => Real.sin (3 * t)', '_',
                    lhs=limit(exp(-x/2)*sin(3*x), x=oo),
                    rhs=0, relation='==', edge=False),
    oracle_instance('2', '0', '_', 'fun _ => 0', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
