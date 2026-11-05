# name: hs-ip-positive-definite-test
# technique: test fallback (Veriphy test_fallback recipe): oracle instances: ∫₀^{2π} |f|² > 0 for continuous f, evaluated exactly
# goal: sage_test KaluzaKlein.HilbertSpaces.InnerProducts.integral_norm_sq_pos file "hs-ip-positive-definite-test.sage"
# description: f = 1 (edge), f = sin x, f = x - π
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

x = var('x')
PI = '(by constructor <;> nlinarith [Real.pi_gt_three])'
cases = [('fun _ : ℝ => (1 : ℂ)', 'continuous_const', '1', PI, 'one_ne_zero', SR(1), True),
         ('fun x : ℝ => (Real.sin x : ℂ)', '(by fun_prop)', '1', PI, '(by simp [Real.sin_pos_of_pos_of_lt_pi, Real.pi_gt_three])', sin(x), False),
         ('fun x : ℝ => ((x - Real.pi : ℝ) : ℂ)', '(by fun_prop)', '1', PI, '(by simp; linarith [Real.pi_gt_three])', x - pi, False)]
instances = [rel([f, hf, x0, hx0, h], integrate(abs(g) ** 2, x, 0, 2 * pi), 0, '>', edge=e) for f, hf, x0, hx0, h, g, e in cases]
