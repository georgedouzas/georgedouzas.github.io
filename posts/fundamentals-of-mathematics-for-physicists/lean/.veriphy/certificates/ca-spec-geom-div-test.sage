# name: ca-spec-geom-div-test
# technique: test fallback (Veriphy test_fallback recipe): kernel-checked instances
# goal: sage_test KaluzaKlein.ComplexAnalysis.AnalyticContinuation.geometric_not_summable file "ca-spec-geom-div-test.sage"
# description: points on and outside the unit circle; Lean decides |z| >= 1 by norm_num through summable_geometric_iff_norm_lt_one.
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
P = 'rw [summable_geometric_iff_norm_lt_one]\nnorm_num'
instances = [kernel_instance('1', '(by norm_num)', proof='rw [summable_geometric_iff_norm_lt_one]\nsimp', edge=True),
             kernel_instance('2', '(by norm_num)', proof=P),
             kernel_instance('-3', '(by norm_num)', proof=P),
             kernel_instance('Complex.I', '(by simp)', proof='rw [summable_geometric_iff_norm_lt_one]\nsimp')]
