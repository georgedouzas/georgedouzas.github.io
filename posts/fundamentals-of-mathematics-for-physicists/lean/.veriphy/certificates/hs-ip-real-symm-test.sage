# name: hs-ip-real-symm-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances: the real inner product of ℝ² is symmetric (and positive on u ≠ 0)
# goal: sage_test KaluzaKlein.HilbertSpaces.InnerProducts.real_inner_symm_pos file "hs-ip-real-symm-test.sage"
# description: symmetry of the dot product at explicit vectors of ℝ² and ℝ, with u = 0 as the edge case
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

cases = [(['EuclideanSpace ℝ (Fin 2)', 'inferInstance', 'inferInstance'], '!₂[(1 : ℝ), 2]', '!₂[(3 : ℝ), -1]', [1, 2], [3, -1], False), (['ℝ', 'inferInstance', 'inferInstance'], '(2 : ℝ)', '(5 : ℝ)', [2], [5], False),
         (['EuclideanSpace ℝ (Fin 2)', 'inferInstance', 'inferInstance'], '!₂[(0 : ℝ), 0]', '!₂[(1 : ℝ), 1]', [0, 0], [1, 1], True)]
instances = [eq(F + [ut, vt], ssum(QQ(a) * QQ(b) for a, b in zip(u, v)), ssum(QQ(b) * QQ(a) for a, b in zip(u, v)), edge=e) for F, ut, vt, u, v, e in cases]
