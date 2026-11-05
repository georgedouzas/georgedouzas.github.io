# name: hs-on-gram-schmidt-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances: Sage runs Gram-Schmidt on the family and compares the Gram matrix of the normalised output with the identity
# goal: sage_test KaluzaKlein.HilbertSpaces.OrthonormalSystems.gramSchmidt_orthonormal_span file "hs-on-gram-schmidt-test.sage"
# description: Gram-Schmidt on the standard bases of ℂ³ and ℂ² and on the empty family (edge)
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

def gs(vs):
    out = []
    for v in vs:
        w = vector(QQbar, v)
        for u in out:
            w = w - (u.conjugate() * w) * u
        out.append(w / sqrt(QQbar(w.conjugate() * w)))
    return out
def gram(es, n):
    return matrix(QQbar, n, n, [[QQbar(a.conjugate() * b) for b in es] for a in es])
cases = [(['EuclideanSpace ℂ (Fin 3)', 'inferInstance', 'inferInstance'], '3', '(fun i => EuclideanSpace.basisFun (Fin 3) ℂ i)', '(EuclideanSpace.basisFun (Fin 3) ℂ).toBasis.linearIndependent', [[1, 0, 0], [0, 1, 0], [0, 0, 1]], False),
         (['EuclideanSpace ℂ (Fin 2)', 'inferInstance', 'inferInstance'], '2', '(fun i => EuclideanSpace.basisFun (Fin 2) ℂ i)', '(EuclideanSpace.basisFun (Fin 2) ℂ).toBasis.linearIndependent', [[1, 0], [0, 1]], False),
         (['EuclideanSpace ℂ (Fin 2)', 'inferInstance', 'inferInstance'], '0', '(![] : Fin 0 → EuclideanSpace ℂ (Fin 2))', 'linearIndependent_empty_type', [], True)]
instances = [oracle_instance(*(E + [n, f, hf]), lhs=gram(gs(vs), len(vs)), rhs=identity_matrix(QQbar, len(vs)), edge=e) for E, n, f, hf, vs, e in cases]
