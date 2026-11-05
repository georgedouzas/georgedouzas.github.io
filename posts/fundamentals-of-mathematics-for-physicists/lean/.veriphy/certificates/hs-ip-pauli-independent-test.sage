# name: hs-ip-pauli-independent-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances: Sage computes the rank of the four matrices (flattened) for trace-orthogonal families
# goal: sage_test KaluzaKlein.HilbertSpaces.InnerProducts.linearIndependent_of_trace_orthogonal file "hs-ip-pauli-independent-test.sage"
# description: the Pauli basis 1, σ₁, σ₂, σ₃ (edge), the same family reordered, and the family 1, -σ₁, iσ₂, σ₃ (the hypothesis is checked by simp)
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

s1 = matrix(QQbar, [[0, 1], [1, 0]]); s2 = matrix(QQbar, [[0, -I], [I, 0]]); s3 = matrix(QQbar, [[1, 0], [0, -1]]); one = identity_matrix(QQbar, 2)
H = '(by intro a b; fin_cases a <;> fin_cases b <;> simp [KaluzaKlein.HilbertSpaces.Basic.pauliFour, KaluzaKlein.HilbertSpaces.Basic.σ1, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3, Matrix.trace, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.conjTranspose_apply] <;> norm_num)'
def rk(ms):
    return ZZ(matrix(QQbar, [list(m.list()) for m in ms]).rank())
fams = [('KaluzaKlein.HilbertSpaces.Basic.pauliFour', [one, s1, s2, s3], True),
        ('![KaluzaKlein.HilbertSpaces.Basic.σ3, KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ1, 1]', [s3, s2, s1, one], False),
        ('![1, -KaluzaKlein.HilbertSpaces.Basic.σ1, Complex.I • KaluzaKlein.HilbertSpaces.Basic.σ2, KaluzaKlein.HilbertSpaces.Basic.σ3]', [one, -s1, I * s2, s3], False)]
instances = [oracle_instance(t, H, lhs=rk(ms), rhs=ZZ(4), edge=e) for t, ms, e in fams]
