# name: ca-spec-arc-ml-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances (rigorous ball integration)
# goal: sage_test KaluzaKlein.ComplexAnalysis.WickRotation.arc_integral_bound file "ca-spec-arc-ml-test.sage"
# description: Sage (arb) integrates over the quarter arc k = R e^(i theta) rigorously and compares |integral| with pi R/(2(R^2 - E^2 - eps)).
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
from sage.all import ComplexBallField as _CBF
CB = _CBF(128)
def arc(E, ep, R):
    E, ep, R = CB(E), CB(ep), CB(R)
    i = CB(0, 1)
    f = lambda th, _: 1/((R*(i*th).exp())**2 - E**2 + ep*i)*(i*R*(i*th).exp())
    return CB.integral(f, 0, CB.pi()/2)
pts = [('0', '0', '1', 0, 0, 1, True), ('1', '1 / 10', '2', 1, QQ(1)/10, 2, False), ('2', '1', '3', 2, 1, 3, False)]
instances = []
for Et, et, Rt, E, ep, R, e in pts:
    v = arc(E, ep, R).abs()
    bound = pi*R/(2*(R**2 - E**2 - ep))
    instances.append(oracle_instance(Et, et, Rt, '(by norm_num)', '(by norm_num)', '(by norm_num)',
                                     lhs=_R(v.lower(), v.upper()), rhs=_ri(_ball(bound).real()), relation='<=', edge=e))
