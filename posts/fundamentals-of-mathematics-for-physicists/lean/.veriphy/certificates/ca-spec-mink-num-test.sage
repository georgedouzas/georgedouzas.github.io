# name: ca-spec-mink-num-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances (Sage integrates exactly, intervals)
# goal: sage_test KaluzaKlein.ComplexAnalysis.SpecialWorked.minkowski_numeric file "ca-spec-mink-num-test.sage"
# description: Sage integrates 1/(k^2 - E^2 + i eps) exactly and checks |Im + pi/E| <= pi eps^2/(2E^5) and |Re| <= pi eps/(2E^3) (the larger ratio <= 1); E = 1, eps = 1e-4 is the proposition, E = 2 the exercise.
SEED = 0

# shared oracle helpers: exact comparison when Sage decides it, else rigorous balls (160 bits)
from sage.all import (SR, ZZ, QQ, I, pi, oo, sqrt, exp, log, sin, cos, gamma, zeta, psi, factorial, bernoulli,
                      harmonic_number, euler_gamma, integrate, limit, var, assume, forget, diff, conjugate, beta,
                      ComplexBallField, RealIntervalField, real_part, imag_part)
from sage.all import sum as ssum
_C = ComplexBallField(160)
_R = RealIntervalField(160)
from sage.rings.complex_arb import ComplexBall as _CB
from sage.rings.real_arb import RealBall as _RB
def _ball(v):
    if isinstance(v, (_CB, _RB)):
        return _C(v)
    return _C(SR(v))
def babs(v):
    """|v| as a rigorous ball (Sage's symbolic abs may drop the absolute value of a complex power)"""
    return _ball(v).abs()
def _ri(rb):
    return _R(rb.lower(), rb.upper())
def eq(values, L, R, edge=False):
    """the claim's two sides at the values: exact when Sage simplifies the difference to 0, else |L - R| in balls"""
    try:
        if isinstance(L, (_CB, _RB)) or isinstance(R, (_CB, _RB)):
            raise ValueError('balls')
        Ls, Rs = SR(L).simplify_full(), SR(R).simplify_full()
        if bool(Ls - Rs == 0):
            return oracle_instance(*values, lhs=Ls, rhs=Rs, edge=edge)
    except Exception:
        pass
    d = (_ball(L) - _ball(R)).abs()
    return oracle_instance(*values, lhs=_ri(d), rhs=_R(0), edge=edge)
def rel(values, L, R, relation, edge=False):
    """a real inequality between the sides, decided on rigorous intervals"""
    return oracle_instance(*values, lhs=_ri(_ball(L).real()), rhs=_ri(_ball(R).real()), relation=relation, edge=edge)
x = var('x')
pts = [('1', '1 / 10000', SR(1), SR(1)/10000, True), ('2', '1 / 10000', SR(2), SR(1)/10000, False), ('1', '1 / 10', SR(1), SR(1)/10, False),
       ('1 / 2', '1 / 100', SR(1)/2, SR(1)/100, False)]
instances = []
for Et, et, E, ep, e in pts:
    v = _ball(integrate(1/(x**2 - E**2 + ep*I), x, -oo, oo))
    r1 = (v.imag() + _ball(pi/E).real()).abs()/_ball(pi*ep**2/(2*E**5)).real()
    r2 = v.real().abs()/_ball(pi*ep/(2*E**3)).real()
    m = r1.max(r2)
    ct = '(' + Et + ' ^ 2 - ' + et + ' * Complex.I) ^ (1 / 2 : ℂ)'
    instances.append(oracle_instance(Et, et, '(by norm_num)', '(by norm_num)', ct, '_', '_', lhs=_ri(m), rhs=_R(1), relation='<=', edge=e))
