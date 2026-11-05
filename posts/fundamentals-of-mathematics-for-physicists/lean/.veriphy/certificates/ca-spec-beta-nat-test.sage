# name: ca-spec-beta-nat-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances (Sage integrates exactly)
# goal: sage_test KaluzaKlein.ComplexAnalysis.BetaFunction.beta_nat file "ca-spec-beta-nat-test.sage"
# description: Sage integrates t^m (1-t)^n over [0, 1] and compares with m! n!/(m + n + 1)!; (1, 2) is B(2, 3) = 1/12, (2, 3) is B(3, 4) = 1/60.
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
t = var('t')
pts = [(0, 0, True), (1, 2, False), (2, 3, False), (4, 1, False)]
instances = [eq([str(m), str(n)], integrate(t**m*(1 - t)**n, t, 0, 1)*factorial(ZZ(m + n + 1)),
                factorial(ZZ(m))*factorial(ZZ(n)), edge=e) for m, n, e in pts]
