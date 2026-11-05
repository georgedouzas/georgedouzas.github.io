# name: ca-spec-finite-part-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances (Sage computes exactly)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ZetaFunction.finite_part_zeta file "ca-spec-finite-part-test.sage"
# description: Sage expands 1/(e^eps - 1) in a Laurent series, reads off the finite part (-1)^k k! [eps^k] of the k-th regularised sum (k = 0..3), checks it is (-1)^k k! B_(k+1)/(k+1)! and compares with zeta(-k).
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
e = var('e')
f = 1/(exp(e) - 1)
def finite_part(k):
    # the eps^0 coefficient of (-d/d eps)^k (1/(e^eps - 1)) is (-1)^k k! times the eps^k coefficient of 1/(e^eps - 1)
    return (-1)**k*factorial(k)*f.series(e, k + 4).coefficient(e, k)
def bern(k):
    return (-1)**k*factorial(k)*bernoulli(k + 1)/factorial(k + 1)
for k in range(4):
    assert finite_part(k) == bern(k), f'the finite part of the regularised sum is not (-1)^k k! B_(k+1)/(k+1)! at k = {k}'
pts = [('0', 0, True), ('1', 1, False), ('2', 2, False), ('3', 3, False)]
instances = [eq([kt], zeta(-ZZ(k)), bern(k), edge=ed) for kt, k, ed in pts]
