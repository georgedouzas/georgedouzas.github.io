# name: hs-ip-cs-identity-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances: ‖u - t v‖² against ‖u‖² - |⟪u, v⟫|²/‖v‖² in ℂ²
# goal: sage_test KaluzaKlein.HilbertSpaces.InnerProducts.norm_sub_proj_sq file "hs-ip-cs-identity-test.sage"
# description: the identity behind Cauchy-Schwarz at explicit vectors, u ∥ v as the edge case (the right side vanishes)
SEED = 0

# shared oracle helpers: exact comparison when Sage decides it, else rigorous balls (160 bits)
from sage.all import (SR, ZZ, QQ, I, pi, oo, sqrt, exp, log, sin, cos, gamma, zeta, factorial, bernoulli, integrate,
                      limit, var, assume, forget, diff, conjugate, matrix, vector, identity_matrix, zero_matrix,
                      ComplexBallField, RealIntervalField, real_part, imag_part, e, binomial, hurwitz_zeta, floor)
from sage.all import sum as ssum
_C = ComplexBallField(160)
_R = RealIntervalField(160)
from sage.rings.complex_arb import ComplexBall as _CB
from sage.rings.real_arb import RealBall as _RB
def _ball(v):
    if isinstance(v, (_CB, _RB)):
        return _C(v)
    return _C(SR(v))
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
def K(values, proof, edge=False):
    return kernel_instance(*values, proof=proof, edge=edge)
def meq(values, L, R, edge=False):
    """exact equality of two Sage matrices or vectors"""
    return oracle_instance(*values, lhs=matrix(L) if not hasattr(L, 'nrows') else L, rhs=matrix(R) if not hasattr(R, 'nrows') else R, edge=edge)

def ip(u, v):
    return ssum(conjugate(SR(a)) * b for a, b in zip(u, v))
def n2(u):
    return ssum(abs(SR(a)) ** 2 for a in u)
def lhs(u, v):
    t = ip(v, u) / n2(v)
    return n2([a - t * b for a, b in zip(u, v)])
HV = '(by simp [EuclideanSpace.ext_iff, Fin.forall_fin_two])'
cases = [('!₂[(1 : ℂ), Complex.I]', '!₂[(Complex.I : ℂ), 2]', [1, I], [I, 2], False), ('!₂[(1 : ℂ), 2]', '!₂[(3 : ℂ), -1]', [1, 2], [3, -1], False),
         ('!₂[(2 : ℂ), 2 * Complex.I]', '!₂[(1 : ℂ), Complex.I]', [2, 2 * I], [1, I], True)]
instances = [eq(['EuclideanSpace ℂ (Fin 2)', 'inferInstance', 'inferInstance'] + [ut, vt, HV], lhs(u, v), n2(u) - abs(ip(u, v)) ** 2 / n2(v), edge=e) for ut, vt, u, v, e in cases]
