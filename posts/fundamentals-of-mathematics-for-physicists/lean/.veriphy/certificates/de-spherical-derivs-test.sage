# name: de-spherical-derivs-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.WaveEquation.spherical_wave_derivs file "de-spherical-derivs-test.sage"
# description: the radial derivatives of f(r - t)/r by Sage for generic f, and f = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
r_, t_ = var('r_ t_'); ff = function('ff')
instances = [
    oracle_instance('_', '_', '_', '_', '_', 'r_', 't_', '_',
                    lhs=diff(ff(r_ - t_)/r_, r_),
                    rhs=diff(ff(r_ - t_), r_)/r_ - ff(r_ - t_)/r_^2, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', 'r_', 't_', '_',
                    lhs=(diff(ff(r_ - t_)/r_, t_, 2) - diff(ff(r_ - t_)/r_, r_, 2) - 2/r_*diff(ff(r_ - t_)/r_, r_)).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun _ => 0', 'fun _ => 0', 'fun _ => 0', '_', '_', '1', '0', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
