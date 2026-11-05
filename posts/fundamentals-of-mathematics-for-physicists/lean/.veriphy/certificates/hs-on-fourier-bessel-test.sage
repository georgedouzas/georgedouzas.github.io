# name: hs-on-fourier-bessel-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances: 2π ∑_{|n| ≤ N} |c_n|² against ∫₀^{2π} |f|², Sage integrating exactly
# goal: sage_test KaluzaKlein.HilbertSpaces.OrthonormalSystems.fourierBessel file "hs-on-fourier-bessel-test.sage"
# description: f = 1 with N = 0 (edge, equality), f = x with N = 1 and N = 3, f = x² with N = 2
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

x = var('x', domain='real')
def c(f, n):
    return integrate(exp(-I * n * x) * f, x, 0, 2 * pi) / (2 * pi)
def lhs(f, N):
    return 2 * pi * ssum(abs(c(f, n)) ** 2 for n in range(-N, N + 1))
cases = [('fun _ : ℝ => (1 : ℂ)', 'continuous_const', '0', SR(1), True), ('fun x : ℝ => (x : ℂ)', '(by fun_prop)', '1', x, False),
         ('fun x : ℝ => (x : ℂ)', '(by fun_prop)', '3', x, False), ('fun x : ℝ => ((x ^ 2 : ℝ) : ℂ)', '(by fun_prop)', '2', x ** 2, False)]
instances = [rel([ft, hf, N, f.subs(x=x)], lhs(f, int(N)), integrate(abs(f) ** 2, x, 0, 2 * pi), '<=', edge=e) for ft, hf, N, f, e in cases]
instances = [rel([ft, hf, N], lhs(f, int(N)), integrate(abs(f) ** 2, x, 0, 2 * pi), '<=', edge=e) for ft, hf, N, f, e in cases]
