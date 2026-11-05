# name: ca-spec-duplication-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances
# goal: sage_test KaluzaKlein.ComplexAnalysis.SpecialWorked.gamma_duplication file "ca-spec-duplication-test.sage"
# description: Sage compares gamma(z) gamma(z + 1/2) with 2^(1-2z) sqrt(pi) gamma(2z) at z = 1, 2, 3, 7/2 (exact) and 3/10 + 7i/10 (balls).
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
pts = [('1', SR(1), True), ('2', SR(2), False), ('3', SR(3), False), ('7 / 2', SR(7)/2, False), ('3 / 10 + 7 / 10 * Complex.I', SR(3)/10 + SR(7)/10*I, False)]
instances = [eq([t], gamma(z)*gamma(z + QQ(1)/2), 2**(1 - 2*z)*sqrt(pi)*gamma(2*z), edge=e) for t, z, e in pts]
