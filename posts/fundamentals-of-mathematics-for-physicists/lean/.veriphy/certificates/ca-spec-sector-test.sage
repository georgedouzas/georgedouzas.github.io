# name: ca-spec-sector-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances (intervals)
# goal: sage_test KaluzaKlein.ComplexAnalysis.WickRotation.re_I_mul_sq_nonpos file "ca-spec-sector-test.sage"
# description: Sage evaluates -a r^2 sin(2 phi) in the sector 0 <= phi <= pi/4.
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
pts = [('1', '1', '0', SR(1), SR(1), SR(0), True), ('2', '3', 'Real.pi / 8', SR(2), SR(3), pi/8, False),
       ('1 / 2', '1', 'Real.pi / 4', SR(1)/2, SR(1), pi/4, False)]
instances = [rel([at, rt, ft, '(by norm_num)', '(by norm_num)', '(by norm_num)'], -a*r**2*sin(2*f), 0, '<=', edge=e)
             for at, rt, ft, a, r, f, e in pts]
