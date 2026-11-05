# name: hs-on-bessel-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances: ∑ |⟪eₖ, u⟫|² against ‖u‖² exactly
# goal: sage_test KaluzaKlein.HilbertSpaces.OrthonormalSystems.bessel file "hs-on-bessel-test.sage"
# description: Bessel's inequality for the standard basis of ℂ² (equality), its first vector (strict), and the empty family (edge)
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
def proj(es, u):
    out = [SR(0)] * len(u)
    for ev in es:
        c = ip(ev, u)
        out = [o + c * a for o, a in zip(out, ev)]
    return out
E1, E2 = [1, 0], [0, 1]
FAMS = [('2', '(fun i => EuclideanSpace.basisFun (Fin 2) ℂ i)', '(EuclideanSpace.basisFun (Fin 2) ℂ).orthonormal', [E1, E2], False), ('1', '(fun _ : Fin 1 => EuclideanSpace.basisFun (Fin 2) ℂ 0)', '((EuclideanSpace.basisFun (Fin 2) ℂ).orthonormal.comp (fun _ : Fin 1 => (0 : Fin 2)) (fun _ _ _ => Subsingleton.elim _ _))', [E1], False), ('0', '(![] : Fin 0 → EuclideanSpace ℂ (Fin 2))', '(by rw [orthonormal_iff_ite]; intro i; exact i.elim0)', [], True)]
U = ('!₂[(1 : ℂ), 2 * Complex.I]', [1, 2 * I])

instances = [rel(['EuclideanSpace ℂ (Fin 2)', 'inferInstance', 'inferInstance'] + [n, e, he, U[0]], ssum(abs(ip(ev, U[1])) ** 2 for ev in es), n2(U[1]), '<=', edge=ed) for n, e, he, es, ed in FAMS]
